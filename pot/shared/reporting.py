"""Shared reporting utilities for experimental reports."""

import time


def print_header(title="PROOF-OF-TRAINING EXPERIMENTAL VALIDATION", framework="PoT Paper Implementation"):
    """Print a standardized report header."""
    print("\n" + "="*80)
    print(f"   {title}")
    print("="*80)
    print(f"\n📊 Report Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"🔬 Framework: {framework}")
    print()


def print_section(title: str, emoji: str = "📌", verbose: bool = True):
    """Print a standardized section header."""
    if verbose:
        print(f"\n{emoji} {title}")
        print("-" * 70)