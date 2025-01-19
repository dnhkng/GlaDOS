import pytest
from glados.spoken_text_converter import SpokenTextConverter

# Create an instance of SpokenTextConverter for testing
converter = SpokenTextConverter()

# Test percentages
@pytest.mark.parametrize("input_text, expected", [
    ("50%", "fifty percent"),
    ("3.5%", "three point five percent"),
    ("100.25%", "one hundred point two five percent"),
    ("0.5%", "zero point five percent"),
    ("1000%", "one thousand percent"),
    ("12.00%", "twelve percent"),
])
def test_convert_percentages(input_text, expected):
    result = converter._convert_percentages(input_text)
    assert result.lower() == expected

# Test times
@pytest.mark.parametrize("input_text, expected", [
    ("3:00", "three o'clock"),
    ("5:05", "five oh five"),
    ("10:30", "ten thirty"),
    ("12:00", "twelve o'clock"),
    ("9:15", "nine fifteen"),
    ("11:45", "eleven forty-five"),
    ("3:00pm", "three o'clock"),
    ("7:30AM", "seven thirty"),
    ("8:05 pm", "eight oh five"),
])
def test_convert_times(input_text, expected):
    result = converter.text_to_spoken(input_text)
    assert result.lower() == expected.lower()

# Test currency
@pytest.mark.parametrize("input_text, expected", [
    ("$1", "one dollar"),
    ("$5.25", "five dollars and twenty-five cents"),
    ("£1", "one pound"),
    ("£1.01", "one pound and one penny"),
    ("$1000", "one thousand dollars"),
    ("$0.50", "zero dollars and fifty cents"),
    ("$1.00", "one dollar"),
    ("£5.00", "five pounds"),
    ("$1000000", "one million dollars"),
    ("$1.99", "one dollar and ninety-nine cents"),
    ("£0.01", "zero pounds and one penny"),
    ("£99.99", "ninety-nine pounds and ninety-nine pence"),
])
def test_convert_currency(input_text, expected):
    result = converter.text_to_spoken(input_text)
    assert result.lower() == expected.lower()

# Test years
@pytest.mark.parametrize("input_text, expected", [
    ("1999", "nineteen ninety-nine"),
    ("2024", "twenty twenty-four"),
    ("2000s", "twenty hundreds"),
    ("1805", "eighteen oh five"),
    ("1900", "nineteen hundred"),
    ("2000", "two thousand"),
    ("1066", "ten sixty-six"),
    ("1776", "seventeen seventy-six"),
    ("2100", "twenty-one hundred"),
    ("1950s", "nineteen fifties"),
])
def test_convert_years(input_text, expected):
    result = converter.text_to_spoken(input_text)
    assert result.lower() == expected.lower()

# Test decimal numbers
@pytest.mark.parametrize("input_text, expected", [
    ("3.14", "three point one four"),
    ("0.5", "zero point five"),
    ("1.0", "one"),
    ("10.01", "ten point zero one"),
    ("100.001", "one hundred point zero zero one"),
    ("0.333", "zero point three three three"),
])
def test_convert_decimal_numbers(input_text, expected):
    result = converter.text_to_spoken(input_text)
    assert result.lower() == expected.lower()

# Test mixed text
@pytest.mark.parametrize("input_text, expected", [
    (
        "The meeting at 3:00pm on 1/1/2024 will cost $50.00.",
        "The meeting at three o'clock on one/one/twenty twenty-four will cost fifty dollars."
    ),
    (
        "In 1999, the company grew by 25% and made £1000000.",
        "In nineteen ninety-nine, the company grew by twenty-five percent and made one million pounds."
    ),
    (
        "Temperature is 98.6° with 0.5% margin of error.",
        "Temperature is ninety-eight point six° with zero point five percent margin of error."
    ),
    (
        "Meeting from 9:00am to 5:00pm costs $100.50.",
        "Meeting from nine o'clock to five o'clock costs one hundred dollars and fifty cents."
    ),
    (
        "8 is the square root of 64",
        "eight is the square root of sixty-four"
    ),
    (
        "8^2 = 64",
        "eight to the power of two equals sixty-four"
    ),
])
def test_convert_mixed_text(input_text, expected):
    result = converter.text_to_spoken(input_text)
    assert result.lower() == expected.lower()

# Test special cases
@pytest.mark.parametrize("input_text, expected", [
    ("Dr. Smith", "Doctor Smith"),
    ("Mr. Jones", "Mister Jones"),
    ("Mrs. Brown", "Mrs Brown"),
    ("Ms. Davis", "Miss Davis"),
    ("100,000", "one hundred thousand"),
    ("1,000,000", "one million"),
    ("yeah", "ye'a"),
])
def test_convert_special_cases(input_text, expected):
    result = converter.text_to_spoken(input_text)
    assert result.lower() == expected.lower()


@pytest.mark.parametrize("input_text, expected", [
    ("8^2 = 64", "eight to the power of two equals sixty-four"),
    ("2^3 = 8", "two to the power of three equals eight"),
    ("√16 = 4", "square root of sixteen equals four"),
    ("∛27 = 3", "cube root of twenty-seven equals three"),
    ("5 + 3 = 8", "five plus three equals eight"),
    ("10 - 5 = 5", "ten minus five equals five"),
    ("4 × 2 = 8", "four times two equals eight"),
    ("8 ÷ 2 = 4", "eight divided by two equals four"),
    ("x^2 + y^2 = z^2", "x to the power of two plus y to the power of two equals z to the power of two"),
    ("15/3 = 5", "fifteen over three equals five"),
])
def test_convert_mathematical_notation(input_text, expected):
    result = converter.text_to_spoken(input_text)
    assert result.lower() == expected.lower()