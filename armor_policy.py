import re
import json
import os
from datetime import datetime
from enum import Enum
from typing import Dict, Any, Optional

class ActionType(Enum):
    """Defines every possible action your bot can take."""
    CLICK = "click"
    FORM_FILL = "form_fill"
    NAVIGATE = "navigate"
    SUBMIT = "submit"
    SCROLL = "scroll"
    DELETE_DATA = "delete_data"
    ACCESS_ADMIN = "access_admin"

class EnforcementResult:
    """Stores the outcome of a policy check."""
    def __init__(self, allowed: bool, reason: str, action_type: str, target: str):
        self.allowed = allowed
        self.reason = reason
        self.action_type = action_type
        self.target = target
        self.timestamp = datetime.now().isoformat()

    def to_dict(self):
        return {
            "allowed": self.allowed,
            "reason": self.reason,
            "action_type": self.action_type,
            "target": self.target,
            "timestamp": self.timestamp
        }

class ArmorPolicyEngine:
    """
    The Policy Engine. It checks every action before it's executed.
    This is your 'Shield'.
    """
    def __init__(self):
        # 1. Allowed Actions (Whitelist)
        self.allowed_actions = {
            ActionType.CLICK,
            ActionType.FORM_FILL,
            ActionType.NAVIGATE,
            ActionType.SUBMIT,
            ActionType.SCROLL,
        }

        # 2. Blocked Patterns (Blacklist)
        #    The bot must never do anything matching these.
        self.blocked_targets = [
            r"delete", r"remove", r"drop",          # Deletion keywords
            r"admin", r"sudo", r"root",             # Admin access
            r"\.\./",                               # Path traversal
            r"/etc/", r"/var/", r"/root/",          # System paths
            r"\.env", r"secret", r"password"        # Secrets
        ]

    def enforce(self, action_type: str, target: str, context: Optional[Dict] = None) -> EnforcementResult:
        """
        The main function that decides if an action is ALLOWED or BLOCKED.
        """
        # Check 1: Is the action type valid and in the allowed list?
        try:
            action_enum = ActionType(action_type.lower())
        except ValueError:
            return EnforcementResult(False, f"Unknown action type: '{action_type}'.", action_type, target)

        if action_enum not in self.allowed_actions:
            return EnforcementResult(False, f"Action '{action_type}' is not allowed by policy.", action_type, target)

        # Check 2: Does the target contain any blocked patterns?
        for pattern in self.blocked_targets:
            if re.search(pattern, target.lower()):
                return EnforcementResult(False, f"Target '{target}' is blocked. Reason: matches pattern '{pattern}'.", action_type, target)

        # Check 3: A specific rule to always block DELETE_DATA actions.
        if action_enum == ActionType.DELETE_DATA:
            return EnforcementResult(False, "DELETE_DATA actions are always blocked by policy.", action_type, target)

        # If all checks pass, the action is allowed.
        return EnforcementResult(True, f"Action '{action_type}' on '{target}' is allowed.", action_type, target)


class ArmorIQAuditLogger:
    """
    Logs every enforcement decision to a JSON Lines file.
    This class is required by bot_script.py.
    """
    def __init__(self, log_path="logs/armor_enforcement.jsonl"):
        self.log_path = log_path
        # Ensure the logs directory exists
        log_dir = os.path.dirname(log_path)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)

    def log(self, result: EnforcementResult):
        """
        Writes an enforcement result to the log file.
        """
        entry = {
            "allowed": result.allowed,
            "reason": result.reason,
            "action_type": result.action_type,
            "target": result.target,
            "timestamp": result.timestamp
        }
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")