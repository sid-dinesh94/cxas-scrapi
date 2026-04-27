# pylint: disable=missing-function-docstring,missing-class-docstring,invalid-name,undefined-variable,line-too-long, broad-exception-caught
"""Unified Member Verification tool for Elevance Polysynth Inbound."""

import datetime
import logging
import re
from typing import Any, Dict


logger = logging.getLogger(__name__)


def unified_member_verification(
    date_of_birth: str,
    member_id: str,
) -> Dict[str, Any]:
  """Backed tool for multiple API calls for member verification.

  A consolidated backend tool that performs date validation, state updates,
  member search,
  policy status evaluation, and reference ID generation in a single execution
  step.

  Args:
      date_of_birth (str): The member's Date of Birth. MUST be strictly formatted as
        'YYYY-MM-DD'. If the user provided an incomplete date, pass
        'incomplete'.
      member_id (str): The alphanumeric Member ID (Health Care ID) collected from the
        user.

  Returns:
      Dict[str, Any]: A dictionary containing the results of the verification
      attempt.
      Expected keys include:
          - 'status' (str): The outcome of the verification. Expected values are
          'success', 'invalid_date_of_birth', 'invalid_member_id', 
          'member_not_found', or 'api_error'.
          - 'next_action' (str): Included only if status is 'success'. Indicates
          the evaluated policy state. Expected values are 'ask_inpatient',
          'secondary_policy', 'multiple_policies', 'collect_dos', or
          'unauthenticated'.
          - 'ai_instruction' (str): Contains instruction hint for the next step.
          - 'message' (str): Error description if the verification failed.
          - 'reference_id_generated' (bool): Indicates if a call reference ID
          was successfully created in the background.
          - 'reference_id' (str): The generated reference ID.
  """
  hcid = "".join(filter(str.isalnum, member_id or "")).upper()
  # ==========================================
  # STEP 1: Validate Date of Birth Format
  # ==========================================
  try:
    parsed_dob = datetime.datetime.strptime(date_of_birth, "%Y-%m-%d").date()
    min_date = datetime.date(1900, 1, 1)
    today = datetime.date.today()
    if parsed_dob <= min_date or parsed_dob > today:
      return {
          "status": "invalid_date_of_birth",
          "message": "Date of birth boundary error.",
      }
  except ValueError:
    return {
        "status": "invalid_date_of_birth",
        "message": "Invalid Date of Birth format. Expected 'YYYY-MM-DD'.",
    }

  # ==========================================
  # STEP 2: Save Base Variables to Persistent Context
  # ==========================================
  try:
    context.state["date_of_birth"] = date_of_birth
    context.state["member_id"] = hcid
  except Exception as e:
    logger.error("Failed to set base context variables: %s", e)

  # ==========================================
  # STEP 3: Validate Member ID
  # ==========================================
  if not hcid or re.fullmatch(r"0+", hcid):
    return {
        "status": "invalid_member_id",
        "message": (
            "Member ID is invalid."
        ),
    }

  # ==========================================
  # STEP 4: Execute Resilient Member Search
  # ==========================================
  conversationId = context.state.get("CallConversationID", "")
  dnisId = context.state.get("DNIS_ID", "")
  override_dnis = context.state.get("OverrideDNIS", "")

  search_payload = {
      "hcid": hcid,
      "birthDate": date_of_birth,
      "conversationId": conversationId,
      "dnisId": dnisId,
      "lob": "COM",  # Hardcoded based on original logic
      "tfn": override_dnis,
      "currentDate": datetime.date.today().strftime("%Y-%m-%d"),
  }

  # EXHAUSTIVE STATE MAPPING (NGA Context Key -> JSON API Key)
  state_mapping = {
      "hipaa_verified": "hipaa_verified",
      "policy_active": "isActive",
      "policy_status": "status",
      "member_groupIdentifier": "groupIdentifier",
      "member_subgroupIdentifier": "subgroupIdentifier",
      "member_source": "source",
      "member_subscriberIdentifier": "subscriberIdentifier",
      "member_firstName": "firstName",
      "member_lastName": "lastName",
      "member_productIdentifier": "productIdentifier",
      "member_contractStateCode": "contractStateCode",
      "member_sequenceNumber": "sequenceNumber",
      "member_dobMatch": "dobMatch",
      "member_coverageEffectiveDate": "coverageEffectiveDate",
      "member_coverageterminationDate": "coverageterminationDate",
      "member_majorBusinessUnitCode": "majorBusinessUnitCode",
      "member_postalCode": "postalCode",
      "member_memberAlphaPrefix": "memberAlphaPrefix",
      "member_routeTypeCode": "RouteTypeCode",
      "member_bhRouteTypeCode": "BHRouteTypeCode",
      "member_mhcRouteTypeCode": "MHCRouteTypeCode",
      "member_isRestricted": "isRestricted",
      "member_multiStatePlan": "multiStatePlan",
      "member_unitCode": "unitCode",
      "member_groupRouteCode": "grpRteCd",
      "member_cdhp": "cdhp",
      "member_gender": "gender",
      "member_coverageTypePhoneNumber": "coverageTypePhoneNumber",
      "member_healthCareArrangementCode": "healthCareArrangementCode",
      "subscriberName": "subscriberName",
      "subscriberDOB": "subscriberDOB",
  }

  # Helper function to handle PolySynth's variable response wrappers
  def extract_payload(response_obj):
    if isinstance(response_obj, dict):
      return response_obj
    if hasattr(response_obj, "json"):
      return response_obj.json()
    raise ValueError("Unrecognized API response format")

  search_success = False
  result = {}

  # Attempt Primary Region
  try:
    resp = tools.memberSearch_verify_member_hipaa(search_payload)
    status_code = getattr(resp, "status_code", 200)  # Default to 200 if missing

    # Accept HTTP 200-499, OR PolySynth's 0/0.0 success code
    # (TODO: quantiphi - only look for 200, don't worry about 0/0.0, or 201-500)
    if (200 <= status_code < 500) or status_code == 0 or status_code == 0.0:
      result = extract_payload(resp)
      search_success = True

    # Failover to Secondary Region
    if not search_success and not result:
      if (200 <= status_code < 500) or status_code == 0 or status_code == 0.0:
        result = extract_payload(resp)
        search_success = True
      else:
        return {
            "status": "api_error",
            "message": f"Secondary API returned HTTP {status_code}",
        }
  except Exception as secondary_error:
    return {
        "status": "api_error",
        "message": f"Member Search Service error: {str(secondary_error)}",
    }

  if not search_success:
    return {
        "status": "api_error",
        "message": "Failed to retrieve member data from all regions.",
    }

  # Evaluate Search Results
  not_found_statuses = [
      "error",
      "unknown",
      "NOT FOUND",
      "No results found for the given Member ID",
      "No matching record found for the entered Date of Birth",
      "No matching record found for the entered DOB",
      "No matching record",
      "No results found",
  ]

  member_dob_found = result.get("dobMatch", False)
  not_found_status = result.get("status", "") in not_found_statuses
  if not_found_status or not member_dob_found:
    return {
        "status": "member_not_found",
        "message": "Identity verification failed. Information did not match.",
    }

  # Sync ALL valid data to persistent context
  for state_key, api_key in state_mapping.items():
    if api_key in result:
      context.state[state_key] = result.get(api_key, "")

  # ==========================================
  # STEP 5: Return Unified Success & Evaluate Policy
  # ==========================================
  policy_stat = str(context.state.get("policy_status", "")).upper()

  next_state = "AUTHENTICATING_MEMBER"
  # Determine the exact next step for the LLM
  if context.state.get("member_isRestricted", False):
    next_action = "transfer_to_agent"
    instruction_for_next_action = (
        "Respond with the following message verbatim: 'I will connect you"
        " with an associate who can help you with your questions.' "
        " DO NOT WAIT FOR USER INPUT and immediately follow the "
        " `agent_transfer_instructions` flow outlined in the main instructions."
    )
    context.state["member_is_verified"] = True
  elif "SECONDARY" in policy_stat:
    next_action = "secondary_policy"
    instruction_for_next_action = (
        "Respond with the following message verbatim: 'Great. The member's"
        " identity is verified. However, our records indicate the member has"
        " another primary insurance. Precertification is not required as a"
        " secondary insurance.' and immediately route to subtask"
        " wrap_menu_not_required_main."
    )
    context.state["member_is_verified"] = True
  elif "MULTIPLE" in policy_stat:
    next_action = "multiple_policies"
    instruction_for_next_action = (
        "Identity verified. The member has multiple active policies. "
        "Follow the subtask flow `multiple_policies_checks` to ask these sequential questions:"
    )
    context.state["member_is_verified"] = True
  elif "INACTIVE" in policy_stat:
    next_action = "collect_dos"
    instruction_for_next_action = (
        "Respond with the following message verbatim: 'Identity verified."
        " However, the policy appears inactive. What is the date of service"
        " for this case?' and immediately route to subtask"
        " collect_date_of_service."
    )
    context.state["member_is_verified"] = True

  elif "TWIN" in policy_stat and "PLUS" in policy_stat:
    next_action = "transfer_to_agent"
    instruction_for_next_action = (
        "Respond with the following message verbatim: 'I will connect you"
        " with an associate who can help you with your questions.' "
        " Follow the `agent_transfer_instructions` flow outlined"
        " in the main instructions."
    )
    context.state["member_is_verified"] = True
  elif "PRIMARY" in policy_stat and "ACTIVE" in policy_stat:
    next_action = "ask_inpatient"  # Standard Active Policy
    next_state = "DETERMINE_COVERAGE"
    instruction_for_next_action = (
        "Respond with the following message verbatim: 'Great. The member's"
        " identity has been verified successfully. Is this for an inpatient or"
        " outpatient service?'"
    )
    context.state["member_is_verified"] = True
  # If the policy status is not one of the expected values, or if the member
  # is not verified, then the member is unauthenticated. (including "HOST",
  # "UNKNOWN", and "COMMERCIAL")
  else:
    next_action = "unauthenticated"
    instruction_for_next_action = (
        "Respond with the following message verbatim: 'I am unable to"
        " authenticate the member's policy. Please call the number on the"
        " back of the member ID Card or return to main menu.' and"
        " immediately route to subtask wrap_menu_not_required_main."
    )
    context.state["member_is_verified"] = False

  # Inject final evaluated state to persistent context
  context.state["current_state"] = next_state

  # ==========================================
  # STEP 6: Auto-Generate Reference ID (Moved and Updated)
  # ==========================================
  ref_code = context.state.get("reference_id", "")
  ref_id_result = None

  is_restricted = context.state.get("member_isRestricted", False)
  is_twin_plus = "TWIN" in policy_stat and "PLUS" in policy_stat
  is_unauthenticated = next_action == "unauthenticated"

  should_call_v1 = (
      not ref_code
      and not is_restricted
      and not is_twin_plus
      and not is_unauthenticated
      and (
          "INACTIVE" in policy_stat
          or "MULTIPLE" in policy_stat
          or "TWIN" in policy_stat
          or "SECONDARY" in policy_stat
          or "PRIMARY" in policy_stat
      )
  )

  if should_call_v1:
    ref_payload = {
        "interactionID": conversationId,
        "providerNPI": context.state.get("npi", ""),
        "MemberID": hcid,
        "memberDOB": date_of_birth,
        "MemberName": (
            f"{context.state.get('member_firstName', '')}"
            f" {context.state.get('member_lastName', '')}".strip()
        ),
        "hipaa": context.state.get("member_is_verified", False),
        "sourceSystem": context.state.get("member_source", ""),
        "ZipCodeMatch": context.state.get("member_postalCode", ""),
        "memberSequenceNumber": context.state.get("member_sequenceNumber", ""),
        "ProductId": context.state.get("member_productIdentifier", ""),
        "DNIS": override_dnis,
        "ANI": context.state.get("ANI", ""),
        "ConversationData": "",
        "EventType": "VirtualAgentCall",
    }

    # Remove empty keys
    ref_payload = {k: v for k, v in ref_payload.items() if v}

    ref_id_result = None
    ref_code = None

    # Attempt Primary
    try:
      ref_resp = tools.reference_id_generate_reference_code(ref_payload)
      primary_result = extract_payload(ref_resp)
      primary_code = primary_result.get(
          "interactionID",
          primary_result.get(
              "reference_code", primary_result.get("referenceId", "")
          ),
      )
      if primary_code:
        ref_id_result = primary_result
        ref_code = primary_code
    except Exception:
      logger.warning("Primary reference ID tool failed.")

    # Attempt Secondary if Primary failed or returned null
    if not ref_code:
      try:
        ref_resp = tools.secondary_reference_id_generate_reference_code(
            ref_payload
        )
        secondary_result = extract_payload(ref_resp)
        secondary_code = secondary_result.get(
            "interactionID",
            secondary_result.get(
                "reference_code", secondary_result.get("referenceId", "")
            ),
        )
        if secondary_code:
          ref_id_result = secondary_result
          ref_code = secondary_code
      except Exception:
        logger.warning("Secondary reference ID tool failed.")

    # Handle Results
    if ref_code:
      context.state["reference_id"] = str(ref_code)
    else:
      # Both failed or returned null, escalate
      next_action = "transfer_to_agent"
      instruction_for_next_action = (
          "Respond with the following message verbatim: 'I will connect you"
          " with an associate who can help you with your questions.'  DO NOT"
          " WAIT FOR USER INPUT and immediately follow the "
          " `agent_transfer_instructions` flow outlined in the main"
          " instructions."
      )

  return {
      "status": "success",
      "state": next_state,
      "next_action": next_action,
      "instruction_for_next_action": instruction_for_next_action,
      "reference_id_generated": bool(ref_id_result),
      "reference_id": str(ref_code) if ref_code else None
  }
