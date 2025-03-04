import json
import os
import logging
import re
import urllib.request
from datetime import datetime
 
# Set up environment variables for Slack integration
SLACK_CHANNEL = os.getenv("SLACK_CHANNEL")
SLACK_TOKEN = os.getenv("SLACK_TOKEN")
 
# Initialize logging
logging.basicConfig(level=logging.INFO)
 
# Constants for user agents and ignored events
USER_AGENTS = {"console.amazonaws.com", "Coral/Jakarta", "Coral/Netty4"}
IGNORED_EVENTS = {
    "DownloadDBLogFilePortion", "TestScheduleExpression", "TestEventPattern",
    "LookupEvents", "listDnssec", "Decrypt",
    "REST.GET.OBJECT_LOCK_CONFIGURATION", "ConsoleLogin",
    "AddCommunicationToCase", "ResolveCase", "CreateCase", "DescribeCases", "CreateSecret", "PutSecretValue", "PutCredentials", "DeleteSession", "CreateSession"
}
 
def check_regex(expr, txt) -> bool:
    return re.search(expr, txt) is not None
 
def match_user_agent(txt) -> bool:
    if txt in USER_AGENTS:
        return True
    expressions = (
        "signin.amazonaws.com(.*)",
        "^S3Console",
        "^Mozilla/",
        "^console(.*)amazonaws.com(.*)",
        "^aws-internal(.*)AWSLambdaConsole(.*)",
    )
    return any(check_regex(expression, txt) for expression in expressions)
 
def match_if_replay(txt) -> bool:
    return check_regex("/replay/", txt)
 
def match_readonly_event_name(txt) -> bool:
    return any(check_regex(expression, txt) for expression in ("^Get", "^Describe", "^List", "^Head"))
 
def match_ignored_events(event_name) -> bool:
    return event_name in IGNORED_EVENTS
 
def extract_email_from_principal_id(principal_id) -> str:
    if ":" in principal_id:
        email_candidate = principal_id.split(":")[1]
        if contains_email(email_candidate):
            return email_candidate
    return None
 
def contains_email(txt) -> bool:
    email_pattern = r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}"
    return bool(re.search(email_pattern, txt))
 
def filter_user_events(event):
    user_agent = event.get('userAgent')
    session_context = event.get('sessionContext', {}).get('sessionIssuer', {})
    
    principal_id = event.get('userIdentity', {}).get('principalId', '')
    extracted_email = extract_email_from_principal_id(principal_id)
 
    if extracted_email:
        logging.info(f"Email extracted from principalId: {extracted_email}")
 
    event_name = event.get('eventName')
 
    is_match = match_user_agent(user_agent) if user_agent else False
    is_replay = match_if_replay(session_context.get('userName', ''))
 
    manual_ec2_events = ["TerminateInstances", "StopInstances", "RebootInstances", "UpdateFunctionCode"]
    
    is_manual_ec2_event = event_name in manual_ec2_events
 
    return (
        is_manual_ec2_event and not is_replay and extracted_email is not None,
        extracted_email,
        principal_id,  # Return principal_id here for later use
        event
    )
 
def send_to_slack(text):
   url = "https://slack.com/api/chat.postMessage"
   payload = json.dumps({
       "channel": SLACK_CHANNEL,
       "text": text
   }).encode('utf-8')
   
   req = urllib.request.Request(url, data=payload, headers={
       'Content-Type': 'application/json',
       'Authorization': f'Bearer {SLACK_TOKEN}'
   })
 
   try:
       with urllib.request.urlopen(req) as response:
           response_body = response.read()
           logging.info(f"Slack response: {response_body.decode('utf-8')}")
           return True  # Indicate success
   except urllib.error.HTTPError as e:
       logging.error(f"HTTPError: {e.code}, {e.read().decode('utf-8')}")
       return False  # Indicate failure
   except Exception as e:
       logging.error(f"Error sending message to Slack: {e}")
       return False  # Indicate failure
 
def lambda_handler(event, context) -> dict:
    if 'Records' in event:
        logging.info("Processing records...")
        for record in event['Records']:
            try:
                raw_body = record['body']
                logging.info(f"Raw SQS message body: {raw_body}")
 
                payload = json.loads(raw_body)
                print(payload)
                if 'Message' in payload:
                    payload = json.loads(payload['Message'])
 
                logging.info(f"Processing payload: {json.dumps(payload, indent=2)}")
 
                output, user_email, principal_id, full_payload = filter_user_events(payload)
                logging.info(f"Output from filter_user_events: {output}, User Email: {user_email}")
 
                session_context = full_payload.get('userIdentity', {}).get('sessionContext', {})
                
                # Extracting User ID from Principal ID (before '@' sign)
                user_id = principal_id.split(':')[1].split('@')[0] if ':' in principal_id else 'N/A'
 
                user_identity_arn = full_payload['userIdentity']['arn']
                event_name = full_payload['eventName']
                if match_ignored_events(event_name):
                    logging.info(f"Skipping event: {event_name} as it's in the IGNORED_EVENTS list.")
                    continue  
                event_time_utc = full_payload['eventTime']
                event_time = datetime.strptime(event_time_utc, "%Y-%m-%dT%H:%M:%SZ")
                event_time_human = event_time.strftime("%A, %B %d, %Y %I:%M:%S %p UTC")
                
                # Extracting the first key-value pair from the requestParameters block dynamically
                request_parameters = full_payload.get('requestParameters', {})
                first_key_value_pair = next(iter(request_parameters.items()), ("N/A", "N/A"))  # Default to 'N/A' if empty
 
                # Construct the message using the first key-value pair
                request_params_message = f"{first_key_value_pair[0]}: {first_key_value_pair[1]}"
                
                ignore_patterns = ["-forwarder", "-fwdr", "-rpy", "-replay"]
                if any(pattern in request_params_message for pattern in ignore_patterns):
                    logging.info(f"Skipping event due to request_params_message containing one of the ignore patterns: {ignore_patterns}")
                    continue
 
                # Check if email pattern exists in principal_id or user_identity_arn
                if contains_email(principal_id) or contains_email(user_identity_arn):
                    message = (
                        f"*Manual AWS Change Detected:*\n"
                        f"*User ID:* `{user_id}`\n"
                        #f"*Principal ID:* `{principal_id}`\n"
                        f"*User Identity ARN:* `{user_identity_arn}`\n"
                        f"*Event:* {event_name}\n"
                        f"*Time:* :clock1: {event_time_human}\n"  # Adding clock emoji next to the human-readable time
                        f"*Resource Info:* ```{request_params_message}```\n"  # Code block for request params
                    )
 
                    send_to_slack(message)
                    logging.info(message)
                    response['body']['events_processed'].append(message)
 
            except json.JSONDecodeError as err:
                logging.exception(f'JSON decode error: {err}')
            except Exception as e:
                logging.exception(f'Error processing record: {e}')
 
