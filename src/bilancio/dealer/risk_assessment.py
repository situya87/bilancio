"""Re-export for backwards compatibility.

RiskAssessor and RiskAssessmentParams have moved to
``bilancio.decision.risk_assessment``.  This shim keeps every existing
import path working.
"""

from bilancio.decision.risk_assessment import RiskAssessmentParams, RiskAssessor

__all__ = ["RiskAssessmentParams", "RiskAssessor"]
