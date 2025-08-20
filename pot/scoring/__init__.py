"""POT Teacher-Forced Scoring Module"""

from .teacher_forced import (
    TeacherForcedScorer,
    OptimizedTeacherForcedScorer,
    FastScorer,
    ScoringConfig,
    ScoringResult,
    create_teacher_forced_challenges
)

# Import from other scoring modules if they exist
try:
    from .optimized_scorer import (
        OptimizedScoringConfig,
    )
except ImportError:
    pass

__all__ = [
    'TeacherForcedScorer',
    'OptimizedTeacherForcedScorer',
    'FastScorer',
    'ScoringConfig',
    'ScoringResult',
    'create_teacher_forced_challenges'
]