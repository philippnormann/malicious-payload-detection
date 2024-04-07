import pandas as pd
import pytest

from src.preprocess import (
    count_payload_len,
    count_special_chars,
    drop_duplicates,
    drop_na,
    html_decode_payload,
    url_decode_payload,
)


@pytest.fixture
def sample_data():
    return pd.DataFrame(
        {
            "payload": ["Test+Payload", "<p>Hello, &amp; World!</p>", "AnotherPayload", "Test+Payload", None],
            "label": ["malicious", "benign", "malicious", "malicious", "benign"],
        }
    )


def test_drop_na(sample_data):
    drop_na(sample_data)
    assert len(sample_data) == 4

    # Check that the None payload is indeed removed
    assert None not in sample_data["payload"].values


def test_drop_duplicates(sample_data):
    drop_duplicates(sample_data)
    assert len(sample_data) == 4

    # Check duplicates are removed
    assert sum(sample_data["payload"] == "Test+Payload") == 1


def test_url_decode_payload():
    assert url_decode_payload("Test+Payload") == "Test Payload"
    assert url_decode_payload("Test%20Payload") == "Test Payload"
    assert url_decode_payload("%3Cscript%3E") == "<script>"


def test_html_decode_payload():
    assert html_decode_payload("<p>Hello, &amp; World!</p>") == "<p>Hello, & World!</p>"
    assert html_decode_payload("&lt;script&gt;") == "<script>"
    assert html_decode_payload("&#60;script&#62;") == "<script>"


def test_count_payload_len(sample_data):
    count_payload_len(sample_data)
    assert "payload_len" in sample_data.columns
    assert sample_data.iloc[0]["payload_len"] == 12  # Length of "Test+Payload"
    assert sample_data.iloc[1]["payload_len"] == 26  # Length of "<p>Hello, &amp; World!</p>"


def test_count_special_chars():
    test_data = pd.DataFrame({"payload": ["<p>Hello, &amp; World!</p>", "!@#$%^&*()", "ABCDE", ""]})
    count_special_chars(test_data)

    assert "special_chars_count" in test_data.columns
    assert test_data.iloc[0]["special_chars_count"] == 9  # Number of special chars in "<p>Hello, &amp; World!</p>"
    assert test_data.iloc[1]["special_chars_count"] == 10  # All characters are special
    assert test_data.iloc[2]["special_chars_count"] == 0  # No special character
    assert test_data.iloc[3]["special_chars_count"] == 0  # Empty string
