
# --- Core Libraries ---
import streamlit as st  # For building the web application UI
import pandas as pd  # For data manipulation, especially with records.xlsx
import random  # For generating random IDs (e.g., member_id, token_number)
from pathlib import Path  # For handling file paths
import json  # For working with JSON data (e.g., from LLM responses)
import re  # For regular expressions (though not explicitly used in this version, good for text processing)
import uuid  # For generating unique identifiers (e.g., report_id)
from datetime import datetime, timedelta  # For handling dates and times (e.g., for test booking)
import os  # For interacting with the operating system, specifically for environment variables

# --- LangGraph Imports ---
from langgraph.graph import StateGraph, END  # Core LangGraph components for defining workflows
from typing import TypedDict, Annotated  # For type hinting, especially for GraphState
from operator import add  # Used with Annotated to append to lists in GraphState

# --- OpenAI and Environment Variable Imports ---
from openai import OpenAI  # OpenAI Python client for interacting with OpenAI API
import dotenv  # For loading environment variables from a .env file
dotenv.load_dotenv()  # Loads environment variables from a .env file into os.environ

# --- PDF Processing Import ---
import PyPDF2  # For extracting text from PDF files

from typing import Optional

# --- API Key Retrieval and OpenAI Client Initialization ---
# Retrieves the OpenAI API key from the environment variables (loaded from .env)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=OPENAI_API_KEY)  # Initializes the OpenAI client with the retrieved API key

# --- Agent Class Definitions ---
# These classes encapsulate the business logic for each specific task in the healthcare workflow.

class MemberVerificationAgent:
    """Handles verification of existing patients or creation of new patient records."""
    def __init__(self, records_file: str = "records.xlsx"):
        # Resolve records.xlsx locally instead of using Google Colab path.
        candidate_paths = [
            Path(records_file),
            Path.cwd() / records_file,
            Path(__file__).parent / records_file,
            Path(__file__).parent / "data" / records_file,
        ]
        self.records_file = None
        for candidate in candidate_paths:
            if candidate.exists():
                self.records_file = candidate.resolve()
                break
        if self.records_file is None:
            self.records_file = (Path(__file__).parent / records_file).resolve()

    def verify_patient(self, phone_or_email: str) -> dict:
        """Verifies patient using phone/email against records.xlsx or creates a new record."""
        if not phone_or_email:
            return {"error": "Please provide a phone number or email address."}

        if not self.records_file.exists():
            return {"error": f"Records file not found at {self.records_file}."}

        records = pd.read_excel(self.records_file)
        records.columns = records.columns.astype(str).str.strip()

        phone_or_email = str(phone_or_email).strip().lower()

        phone_col = "Phone_number" if "Phone_number" in records.columns else "Phone"
        email_col = "Email" if "Email" in records.columns else "email"

        if phone_col not in records.columns or email_col not in records.columns:
            return {"error": f"records.xlsx must contain Phone_number and Email columns. Found columns: {list(records.columns)}"}

        matches = records[
            records[phone_col].astype(str).str.strip().str.lower().eq(phone_or_email)
            | records[email_col].astype(str).str.strip().str.lower().eq(phone_or_email)
        ]

        if len(matches) > 0:
            patient_record = matches.iloc[0].to_dict()

            return {
                "member_id": f"M{matches.index[0] + 100000}",
                "member_type": "Existing",
                "patient_record": patient_record,
            }

        # If no match found, create a new member ID and a placeholder record.
        new_member_id = f"M{random.randint(100000, 999999)}"  # Generates a random new member ID

        patient_record = {
            "Phone_number": phone_or_email if "@" not in phone_or_email else None,
            "Email": phone_or_email if "@" in phone_or_email else None,
            "Name": None,
            "Age": None,
            "Gender": None,
            "Address": None,
            "Summary": None,
        }

        return {
            "member_id": new_member_id,
            "member_type": "New",
            "patient_record": patient_record,
        }


class ReportManagementAgent:
    """Manages the linking and storage of patient medical reports."""
    def __init__(self):
        # Initializes an empty in-memory registry for reports.
        self.report_registry = {}

    def link_report(self, member_id: str, report_path: str) -> dict:
        """Links a given report path to a member ID and checks its existence."""
        st.write(f"[DEBUG ReportManagementAgent] Received member_id: {member_id}, report_path: {report_path}")

        if not report_path:
            st.write("[DEBUG ReportManagementAgent] report_path is empty. Returning 'Missing' status.")
            return {
                "report_id": None,
                "report_status": "Missing"
            }

        path = Path(report_path)
        st.write(f"[DEBUG ReportManagementAgent] Path object created: {path}")

        if not path.exists():
            st.write(f"[DEBUG ReportManagementAgent] Path does NOT exist: {path}. Returning 'Error' status.")
            return {
                "report_id": None,
                "report_status": "Error",
                "error": "Report file could not be found."
            }

        st.write(f"[DEBUG ReportManagementAgent] Path EXISTS: {path}.")
        report_id = f"R{uuid.uuid4().hex[:8].upper()}"  # Generate a unique report ID

        # Stores the report details in the registry.
        self.report_registry[report_id] = {
            "member_id": member_id,
            "report_path": report_path
        }
        st.write(f"[DEBUG ReportManagementAgent] Report linked successfully. Report ID: {report_id}")

        return {
            "report_id": report_id,
            "report_status": "Uploaded"
        }


class DiagnosticSummaryAgent:
    """Generates a plain-language summary of diagnostic reports using an LLM."""
    def summarize_report(self, member_id: str, report_text: str) -> dict:
        """Sends report text to an LLM for summarization and extraction of key findings."""
        if not report_text:
            return {
                "summary": "No report was available to summarize.",
                "abnormal_findings": [],
                "recommended_follow_up": [],
            }

        # Prompt engineered to instruct the LLM on summarization and safety rules
        prompt = f"""
You are a healthcare assistant helping explain diagnostic report findings.

Member ID: {member_id}

Report Text:
{report_text}

Return a JSON object with:
- summary: plain-language summary
- abnormal_findings: list of abnormal or out-of-range findings
- recommended_follow_up: list of safe follow-up suggestions

Rules:
- Do not diagnose disease.
- Do not prescribe medication.
- Do not invent missing medical details.
- If values are unclear, say they could not be interpreted.
- Recommend discussing abnormal results with a clinician when appropriate.
"""

        # If OPENAI_API_KEY is not available, use a safe local fallback so the UI still responds.
        if not OPENAI_API_KEY:
            short_text = report_text[:800].replace("\n", " ")
            return {
                "summary": f"Report uploaded and processed. Key extracted text: {short_text}",
                "abnormal_findings": [],
                "recommended_follow_up": ["Please review the report with a qualified clinician."]
            }

        # Calls the OpenAI chat completions API
        try:
            response = client.chat.completions.create( 
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                response_format={"type": "json_object"}
            )
            return json.loads(response.choices[0].message.content)
        except Exception as e:
            return {
                "summary": f"Report was processed, but AI summarization failed: {e}",
                "abnormal_findings": [],
                "recommended_follow_up": ["Please review this summary with a qualified clinician."]
            }


class TestBookingAgent:
    """Manages the booking of medical tests for patients."""
    VALID_TESTS = {  # Defines a set of valid tests that can be booked
        "cholesterol test",
        "blood glucose test",
        "cbc",
        "lipid panel",
        "thyroid panel",
    }

    def book_test(
        self,
        member_id: str,
        requested_test: str,
        preferred_date: Optional[str] = None
    ) -> dict:
        """Books a specific test for a member on a preferred or next available date."""

        if not requested_test:
            return {
                "booking_status": "Failed",
                "message": "Please tell me which test you would like to book."
            }

        requested_test = requested_test.lower().strip()

        if requested_test not in self.VALID_TESTS:
            return {
                "booking_status": "Failed",
                "message": f"'{requested_test}' is not an available test."
            }

        appointment_time = self._find_available_slot(preferred_date)  # Finds an available slot

        if appointment_time is None:
            return {
                "booking_status": "Failed",
                "message": "No appointment slots are currently available."
            }

        token_number = f"T{random.randint(10000, 99999)}"  # Generates a random token number

        return {
            "booking_status": "Confirmed",
            "token_number": token_number,
            "appointment_time": appointment_time,
        }

    def _find_available_slot(self, preferred_date):
        """STUB function to simulate finding an available appointment slot."""
        if preferred_date:
            return f"{preferred_date} 10:30 AM"  # If a preferred date is given, use it

        tomorrow = datetime.now() + timedelta(days=1)
        return tomorrow.strftime("%Y-%m-%d 10:30 AM")  # Otherwise, suggest tomorrow


class FinalResponseAgent:
    """Generates a personalized, professional closing message for the patient."""
    def generate_response(
        self,
        member_id: str,
        diagnostic_summary: str,
        booking_details: Optional[dict] = None
    ) -> str:
        """Composes a final message including summary and booking details (if available)."""

        booking_text = ""

        if booking_details:
            # Formats booking details if they exist
            booking_text = f"""
Booking Details:
- Token Number: {booking_details.get('token_number')}
- Appointment Time: {booking_details.get('appointment_time')}
"""

        # Prompt engineered to guide the LLM in crafting the final message
        prompt = f"""
Create a final closing message for a healthcare assistant.

Member ID:
{member_id}

Diagnostic Summary:
{diagnostic_summary}

{booking_text}

Requirements:
- Thank the patient
- Keep the tone clear and professional
- Include booking information if provided
- Do not invent appointment details.
- Keep the message concise
"""

        # If OPENAI_API_KEY is not available, use a local fallback so the button always returns output.
        if not OPENAI_API_KEY:
            message = f"Thank you. Your member ID is {member_id}.\n\nSummary: {diagnostic_summary}"
            if booking_details:
                message += f"\n\nBooking confirmed. Token Number: {booking_details.get('token_number')}. Appointment Time: {booking_details.get('appointment_time')}."
            return message

        # Calls the OpenAI chat completions API
        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            message = f"Thank you. Your member ID is {member_id}.\n\nSummary: {diagnostic_summary}\n\nAI final response generation failed: {e}"
            if booking_details:
                message += f"\n\nBooking confirmed. Token Number: {booking_details.get('token_number')}. Appointment Time: {booking_details.get('appointment_time')}."
            return message

# --- PDF Text Extraction Function ---
def extract_report_text(report_path: str) -> str:
    """Extracts text content from a PDF file located at report_path."""
    st.write(f"[DEBUG extract_report_text] Received report_path: {report_path}")
    path = Path(report_path)
    st.write(f"[DEBUG extract_report_text] Path object: {path}")

    if not path.exists():
        st.write(f"[DEBUG extract_report_text] Path does NOT exist: {path}")
        return ""

    st.write(f"[DEBUG extract_report_text] Path EXISTS: {path}")
    text_parts = []

    try:
        with open(path, "rb") as f:
            reader = PyPDF2.PdfReader(f)

            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text_parts.append(page_text)
        st.write(f"[DEBUG extract_report_text] Successfully extracted text. Length: {len(''.join(text_parts))}")
    except Exception as e:
        st.write(f"[DEBUG extract_report_text] Error during PDF extraction: {e}")
        return ""  # Return empty string on extraction error

    return "\n".join(text_parts).strip()  # Joins all extracted page texts into a single string


# --- Patient-to-Report Matching Helpers ---
def normalize_value(value) -> str:
    """Normalizes text for matching phone, email, name, and DOB values."""
    if value is None:
        return ""
    if pd.isna(value):
        return ""
    return str(value).strip().lower()


def normalize_phone(value) -> str:
    """Keeps only digits so +91-98450-11223 and 9845011223 can match."""
    return re.sub(r"\D", "", normalize_value(value))


def get_patient_field(patient_record: dict, possible_names: list) -> str:
    """Returns a patient field from records.xlsx using flexible column names."""
    if not patient_record:
        return ""
    lowered = {str(k).strip().lower(): v for k, v in patient_record.items()}
    for name in possible_names:
        key = name.strip().lower()
        if key in lowered:
            return normalize_value(lowered[key])
    return ""


def report_matches_patient(report_path: str, phone_or_email: str, patient_record: dict) -> tuple:
    """
    Checks whether a PDF report belongs to the verified patient.

    Matching priority:
    1. Exact submitted phone number found in report text
    2. Exact submitted email found in report text
    3. Phone number from records.xlsx found in report text
    4. Email from records.xlsx found in report text
    5. Patient name from records.xlsx found in report text
    6. DOB from records.xlsx found in report text
    """
    report_text = extract_report_text(report_path)
    report_text_lower = report_text.lower()
    report_digits = normalize_phone(report_text)

    submitted = normalize_value(phone_or_email)
    submitted_digits = normalize_phone(phone_or_email)

    if submitted_digits and len(submitted_digits) >= 7 and submitted_digits in report_digits:
        return True, "matched submitted phone number", report_text

    if "@" in submitted and submitted in report_text_lower:
        return True, "matched submitted email", report_text

    record_phone = get_patient_field(patient_record, ["Phone_number", "Phone", "Mobile", "Contact", "Phone Number"])
    record_email = get_patient_field(patient_record, ["Email", "email", "Email Address"])
    record_name = get_patient_field(patient_record, ["Name", "Patient", "Patient Name", "Patient_Name"])
    record_dob = get_patient_field(patient_record, ["DOB", "Date of Birth", "Date_of_birth", "Birth Date"])

    record_phone_digits = normalize_phone(record_phone)

    if record_phone_digits and len(record_phone_digits) >= 7 and record_phone_digits in report_digits:
        return True, "matched phone number from records.xlsx", report_text

    if record_email and "@" in record_email and record_email in report_text_lower:
        return True, "matched email from records.xlsx", report_text

    if record_name and record_name in report_text_lower:
        return True, "matched patient name from records.xlsx", report_text

    if record_dob:
        dob_variants = {
            record_dob,
            record_dob.replace("-", "/"),
            record_dob.replace("/", "-"),
        }
        for dob in dob_variants:
            if dob and dob in report_text_lower:
                return True, "matched DOB from records.xlsx", report_text

    return False, "no exact patient identifier matched", report_text


def find_matching_local_report(project_dir: Path, phone_or_email: str, patient_record: dict) -> tuple:
    """
    Finds the correct local PDF report for the submitted phone/email.
    Does NOT use the first PDF blindly.
    """
    pdfs = sorted(project_dir.glob("*.pdf"))

    for pdf_path in pdfs:
        matched, reason, report_text = report_matches_patient(str(pdf_path), phone_or_email, patient_record)
        if matched:
            return str(pdf_path), reason, report_text

    return None, "No local PDF matched the submitted phone/email or verified patient record.", ""



# --- GraphState Definition ---
class GraphState(TypedDict):
    """
    Represents the state of our LangGraph. This TypedDict defines the schema for data that 
    flows between nodes in the workflow. Each node updates this state.
    """
    phone_or_email: Optional[str]
    report_path: Optional[str]
    requested_test: Optional[str]
    preferred_date: Optional[str]
    member_id: Optional[str]
    member_type: Optional[str]
    patient_record: Optional[dict]
    report_id: Optional[str]
    report_status: Optional[str]
    report_text: Optional[str]  # Extracted report text
    diagnostic_summary: Optional[str]
    abnormal_findings: Optional[list]
    recommended_follow_up: Optional[list]
    booking_status: Optional[str]
    token_number: Optional[str]
    appointment_time: Optional[str]
    final_message: Optional[str]
    workflow_status: Optional[str]  # 'Waiting for Input', 'Completed', 'Failed'
    logs: Annotated[list[str], add]  # Uses 'add' operator to append new log entries to the list
    error: Optional[str]  # General error message for workflow failures
    next_agent: Optional[str]  # Used for explicit routing decisions between nodes


# --- LangGraph Node Functions ---
# These functions wrap the logic of the agent classes and update the GraphState.

def member_verification_node(state: GraphState) -> GraphState:
    """Node for member verification. Calls MemberVerificationAgent to verify or create a patient record."""
    logs = state.get('logs', [])
    logs.append("Executing Member Verification Node")

    phone_or_email = state.get("phone_or_email")

    if not phone_or_email:
        # If input is missing, sets workflow status to 'Waiting for Input'
        return {
            "workflow_status": "Waiting for Input",
            "final_message": "Please provide your phone number or email address so I can verify your membership.",
            "logs": logs,
            "next_agent": "final_response_node"  # Routes to final response to ask for input
        }

    verification_result = member_agent.verify_patient(phone_or_email)
    logs.append("Called MemberVerificationAgent.verify_patient")

    if "error" in verification_result:
        # If verification fails, sets workflow status to 'Failed' with the error message.
        return {
            "workflow_status": "Failed",
            "final_message": verification_result["error"],
            "logs": logs,
            "error": verification_result["error"],
            "next_agent": "final_response_node"  # Routes to final response on error
        }

    # Updates state with member details and sets next agent to report management.
    return {
        "member_id": verification_result["member_id"],
        "member_type": verification_result["member_type"],
        "patient_record": verification_result["patient_record"],
        "logs": logs,
        "next_agent": "report_management_node" 
    }

def report_management_node(state: GraphState) -> GraphState:
    """Node for report management. Calls ReportManagementAgent to link reports and extracts text content from PDFs."""
    logs = state.get('logs', [])
    logs.append("Executing Report Management Node")

    member_id = state.get("member_id")
    report_path = state.get("report_path")
    updated_state = {"logs": logs, "next_agent": "diagnostic_summary_node"}

    if report_path and member_id:
        # Extracts text from PDF report using the helper function.
        report_text = extract_report_text(report_path)
        updated_state["report_text"] = report_text

        report_result = report_agent.link_report(member_id, report_path)
        logs.append("Called ReportManagementAgent.link_report")

        if report_result.get("report_status") == "Error":
            # If report linking fails, sets error and bypasses summary, moving to test booking.
            updated_state["error"] = report_result.get("error")
            updated_state["diagnostic_summary"] = "The report could not be summarized due to an error during linking."
            updated_state["next_agent"] = "test_booking_node" 
        else:
            # Updates state with successful report linking details.
            updated_state["report_id"] = report_result.get("report_id")
            updated_state["report_status"] = report_result.get("report_status")
    elif not report_path:
        # If no report path provided, skips report management and summary.
        logs.append("No report path provided, skipping report management.")
        updated_state["diagnostic_summary"] = "No report was available for summarization."
        updated_state["next_agent"] = "test_booking_node" 

    return updated_state

def diagnostic_summary_node(state: GraphState) -> GraphState:
    """Node for diagnostic summary. Calls DiagnosticSummaryAgent to summarize the extracted report text."""
    logs = state.get('logs', [])
    logs.append("Executing Diagnostic Summary Node")

    member_id = state.get("member_id")
    report_text = state.get("report_text")
    updated_state = {"logs": logs, "next_agent": "test_booking_node"}

    if member_id and report_text:
        # Calls the DiagnosticSummaryAgent to summarize the report.
        summary_result = symmary_agent.summarize_report(
            member_id=member_id,
            report_text=report_text,
        )
        logs.append("Called DiagnosticSummaryAgent.summarize_report")

        # Updates state with the diagnostic summary and extracted findings.
        updated_state["diagnostic_summary"] = summary_result.get("summary")
        updated_state["abnormal_findings"] = summary_result.get("abnormal_findings", [])
        updated_state["recommended_follow_up"] = summary_result.get("recommended_follow_up", [])
    else:
        # If no report text, skips summarization and ensures a default summary message.
        logs.append("No report text or member_id for summarization, skipping diagnostic summary.")
        if not updated_state.get("diagnostic_summary"):
            updated_state["diagnostic_summary"] = "No report was available for summarization."

    return updated_state

def test_booking_node(state: GraphState) -> GraphState:
    """Node for test booking. Calls TestBookingAgent to book a requested medical test."""
    logs = state.get('logs', [])
    logs.append("Executing Test Booking Node")

    member_id = state.get("member_id")
    requested_test = state.get("requested_test")
    preferred_date = state.get("preferred_date")
    updated_state = {"logs": logs, "next_agent": "final_response_node"}

    if requested_test and member_id:
        # Calls the TestBookingAgent to book a test.
        booking_result = booking_agent.book_test(
            member_id=member_id,
            requested_test=requested_test,
            preferred_date=preferred_date,
        )
        logs.append("Called TestBookingAgent.book_test")

        if booking_result.get("booking_status") == "Confirmed":
            # Updates state with confirmed booking details.
            updated_state["booking_status"] = booking_result.get("booking_status")
            updated_state["token_number"] = booking_result.get("token_number")
            updated_state["appointment_time"] = booking_result.get("appointment_time")
        else:
            # If booking fails, records the error message.
            updated_state["error"] = booking_result.get("message", "Test booking failed.")
            logs.append(f"Test booking failed: {updated_state['error']}")
    else:
        # If no test requested or member ID missing, skips booking.
        logs.append("No test requested or member_id missing, skipping test booking.")

    return updated_state

def final_response_node(state: GraphState) -> GraphState:
    """Node for generating the final patient-facing response. Calls FinalResponseAgent."""
    logs = state.get('logs', [])
    logs.append("Executing Final Response Node")

    member_id = state.get("member_id")
    diagnostic_summary = state.get("diagnostic_summary", "No diagnostic summary available.")

    booking_details = None
    if state.get("booking_status") == "Confirmed":
        # Prepares booking details for the final response agent.
        booking_details = {
            "token_number": state.get("token_number"),
            "appointment_time": state.get("appointment_time"),
        }

    # Calls the FinalResponseAgent to generate the concluding message.
    final_message = response_agent.generate_response(
        member_id=member_id,
        diagnostic_summary=diagnostic_summary,
        booking_details=booking_details,
    )
    logs.append("Called FinalResponseAgent.generate_response")

    # Determines the final workflow status based on any errors or pending input.
    workflow_status = "Completed"
    if state.get("error"):
        workflow_status = "Failed"
    elif state.get("workflow_status") == "Waiting for Input":
        workflow_status = "Waiting for Input"

    # Returns the final message and status, signaling the end of the workflow.
    return {
        "final_message": final_message,
        "workflow_status": workflow_status,
        "logs": logs,
        "next_agent": END  # Signals to LangGraph that this is the end node
    }

# --- Workflow Router (for potential conditional routing, currently direct edges are used) ---
def route_workflow(state: GraphState) -> str:
    """Determines the next node based on the current state (e.g., errors, missing input)."""
    if state.get("error"):
        return "final_response_node"
    if state.get("workflow_status") == "Waiting for Input":
        return "final_response_node"
    if state.get("next_agent"):
        return state["next_agent"]
    return "final_response_node"  # Default routing to final response if no specific next_agent is set


# --- Agent Instantiation ---
# Creates instances of each agent class, which will be used by the LangGraph nodes.
# records.xlsx needs to be accessible, hence the absolute path.
member_agent = MemberVerificationAgent(records_file="/Users/pramodkhadake/Documents/AIProject_HA/records.xlsx")
report_agent = ReportManagementAgent()
symmary_agent = DiagnosticSummaryAgent()
booking_agent = TestBookingAgent()
response_agent = FinalResponseAgent()

# --- LangGraph Building and Compilation ---
# Defines the structure and flow of the multi-agent workflow.

workflow = StateGraph(GraphState)  # Initializes the StateGraph with our defined GraphState

# Adds each node function to the workflow, associating a name with each function.
workflow.add_node("member_verification_node", member_verification_node)
workflow.add_node("report_management_node", report_management_node)
workflow.add_node("diagnostic_summary_node", diagnostic_summary_node)
workflow.add_node("test_booking_node", test_booking_node)
workflow.add_node("final_response_node", final_response_node)

workflow.set_entry_point("member_verification_node")  # Sets the starting point of the workflow execution.

# Defines the sequential edges (transitions) between nodes, establishing the main workflow path.
workflow.add_edge("member_verification_node", "report_management_node")
workflow.add_edge("report_management_node", "diagnostic_summary_node")
workflow.add_edge("diagnostic_summary_node", "test_booking_node")
workflow.add_edge("test_booking_node", "final_response_node")

workflow.set_finish_point("final_response_node")  # Sets the node that signals the end of the workflow execution.

app = workflow.compile()  # Compiles the graph into an executable LangGraph application.


# --- Streamlit UI ---
# This section defines the user interface for the healthcare assistant.

st.set_page_config(layout="wide")  # Configures the Streamlit page layout for wider content.
st.title("AI-Powered Healthcare Assistant")  # Sets the main title of the Streamlit application.

st.markdown("---")
st.markdown("### Patient Information")

# Input field for patient's phone number or email for verification.
phone_or_email = st.text_input("Enter Phone Number or Email for Verification", key="phone_email_input")

st.markdown("### Medical Report")
# File uploader widget for PDF medical reports.
report_file = st.file_uploader("Upload Medical Report (PDF)", type=["pdf"], key="report_uploader")

st.markdown("### Test Booking (Optional)")
# Input fields for test booking requests.
requested_test = st.text_input("Requested Test (e.g., 'cholesterol test', 'blood glucose test')", key="test_input")
preferred_date = st.date_input("Preferred Date for Test (Optional)", value=None, key="date_input")

# --- Button to trigger the workflow ---
if st.button("Run Healthcare Workflow", key="run_button"):
    if not phone_or_email:
        st.error("Please provide a phone number or email for verification.")
    else:
        uploaded_temp_file = None
        report_path_for_graph = None

        # Verify patient first so report selection can use the matched patient record.
        verification_preview = member_agent.verify_patient(phone_or_email)

        if "error" in verification_preview:
            st.error(verification_preview["error"])
            st.stop()

        verified_patient_record = verification_preview.get("patient_record", {})
        st.success(
            f"Patient verification complete: {verification_preview.get('member_type')} "
            f"member {verification_preview.get('member_id')}"
        )

        if report_file:
            # Option 1: use the PDF uploaded in Streamlit.
            temp_dir = Path("/tmp")
            temp_dir.mkdir(exist_ok=True)

            temp_report_path = temp_dir / report_file.name

            with open(temp_report_path, "wb") as f:
                f.write(report_file.getvalue())

            uploaded_temp_file = temp_report_path
            matched, match_reason, _ = report_matches_patient(
                str(temp_report_path),
                phone_or_email,
                verified_patient_record
            )

            if not matched:
                st.error(
                    "The uploaded report does not match the submitted phone/email "
                    "or the verified patient record. Please upload the correct patient's report."
                )
                st.warning(f"Match result: {match_reason}")
                st.stop()

            report_path_for_graph = str(temp_report_path)
            st.info(f"Using uploaded report: {report_path_for_graph}")
            st.success(f"Report match confirmed: {match_reason}")

        else:
            # Option 2: auto-select a local PDF ONLY if it matches the submitted patient.
            project_dir = Path("/Users/pramodkhadake/Documents/AIProject_HA")

            report_path_for_graph, match_reason, _ = find_matching_local_report(
                project_dir=project_dir,
                phone_or_email=phone_or_email,
                patient_record=verified_patient_record
            )

            if report_path_for_graph:
                st.info(f"Using matched local report: {report_path_for_graph}")
                st.success(f"Report match confirmed: {match_reason}")
            else:
                st.warning(
                    "No local PDF report matched the submitted phone/email or verified patient record. "
                    "The workflow will continue without a report summary. Upload the matching report "
                    "or make sure the PDF contains the patient's phone, email, name, or DOB."
                )

        # --- Initialize the LangGraph state with user inputs ---
        initial_state_langgraph = GraphState(
            phone_or_email=phone_or_email,
            report_path=report_path_for_graph,
            requested_test=requested_test if requested_test else None,
            preferred_date=str(preferred_date) if preferred_date else None,
            logs=[],
            error=None,
            next_agent=None,
            member_id=None,
            member_type=None,
            patient_record=None,
            report_id=None,
            report_status=None,
            report_text=None,
            diagnostic_summary=None,
            abnormal_findings=None,
            recommended_follow_up=None,
            booking_status=None,
            token_number=None,
            appointment_time=None,
            final_message=None,
            workflow_status=None
        )

        # --- Execute the LangGraph workflow ---
        with st.spinner("Running healthcare workflow..."):
            final_accumulated_state = app.invoke(initial_state_langgraph)

        # --- Display Workflow Results in Streamlit UI ---
        st.markdown("---")
        st.subheader("Workflow Results")

        st.markdown(f"**Workflow Status:** `{final_accumulated_state['workflow_status']}`")

        st.info(f"**Final Message:** {final_accumulated_state['final_message']}")

        if final_accumulated_state.get('error'):
            st.error(f"Error: {final_accumulated_state['error']}")

        st.subheader("Detailed State")
        st.json(final_accumulated_state)

        st.subheader("Workflow Logs")
        for i, log_entry in enumerate(final_accumulated_state["logs"]):
            st.text(f"{i+1}. {log_entry}")

        # Clean up only uploaded temp files. Do NOT delete local project PDFs.
        if uploaded_temp_file and uploaded_temp_file.exists():
            uploaded_temp_file.unlink()
            st.write(f"Cleaned up temporary uploaded file: {uploaded_temp_file}")

