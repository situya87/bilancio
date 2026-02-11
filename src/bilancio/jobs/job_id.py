"""Job ID generation for memorable simulation job identifiers."""

from xkcdpass import xkcd_password as xp

# Use the EFF large wordlist (7776 words, designed for memorability)
_wordlist = xp.generate_wordlist(wordfile=xp.locate_wordfile())


def generate_job_id(num_words: int = 4) -> str:
    """Generate a memorable job ID as a hyphen-separated passphrase.

    Uses the EFF wordlist via xkcdpass for high-quality memorable words.

    Args:
        num_words: Number of words in the passphrase (default 4)

    Returns:
        A job ID like "castle-river-mountain-forest"

    Example:
        >>> job_id = generate_job_id()
        >>> print(job_id)  # "correct-horse-battery-staple"
    """
    return str(xp.generate_xkcdpassword(_wordlist, numwords=num_words, delimiter="-"))


def validate_job_id(job_id: str) -> bool:
    """Validate that a job ID has the correct format.

    Args:
        job_id: The job ID to validate

    Returns:
        True if valid format (hyphen-separated words), False otherwise
    """
    if not job_id:
        return False
    parts = job_id.split("-")
    if len(parts) < 2:  # At least 2 words
        return False
    return all(part.isalpha() and part.islower() for part in parts)
