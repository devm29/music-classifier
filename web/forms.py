"""Validation of the one thing this app accepts from the outside world."""

from __future__ import annotations

from pathlib import Path

from django import forms

from classifier.audio import MAX_UPLOAD_BYTES, SUPPORTED_SUFFIXES


class UploadForm(forms.Form):
    """
    An audio upload, checked before a single byte is decoded.

    Both checks run here rather than in the decoder because rejecting a 2 GB
    file after writing it to disk and handing it to ffmpeg is not rejecting it.
    """

    file = forms.FileField()

    def clean_file(self):
        upload = self.cleaned_data["file"]
        suffix = Path(upload.name).suffix.lower()
        if suffix not in SUPPORTED_SUFFIXES:
            raise forms.ValidationError(
                f"{suffix or 'that'} is not a supported audio format. "
                f"Supported: {', '.join(SUPPORTED_SUFFIXES)}."
            )
        if upload.size == 0:
            raise forms.ValidationError("the uploaded file is empty.")
        if upload.size > MAX_UPLOAD_BYTES:
            raise forms.ValidationError(
                f"the file is {upload.size / 1_048_576:.0f} MB; the limit is "
                f"{MAX_UPLOAD_BYTES // 1_048_576} MB."
            )
        return upload
