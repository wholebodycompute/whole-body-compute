#!/usr/bin/env python3
"""
Backward-compatible alias for `whole_body_compute.chaos`.
"""

from __future__ import annotations

from whole_body_compute.chaos import ChaosOrganFuzzer, main

__all__ = ["ChaosOrganFuzzer", "main"]


if __name__ == "__main__":
    main()
