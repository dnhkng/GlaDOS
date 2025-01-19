import re


class SpokenTextConverter:
    """
    A utility class for converting text containing numbers, dates, times, and currency
    into their spoken-word equivalents. Can you inagine how many edge cases you have to cover?

    This class provides methods to normalize and convert various text elements, such as:
    - Numbers (e.g., "3.14" → "three point one four")
    - Dates (e.g., "1/1/2024" → "one/one/twenty twenty-four")
    - Times (e.g., "3:00pm" → "three o'clock")
    - Currency (e.g., "$50.00" → "fifty dollars")
    - Percentages (e.g., "50%" → "fifty percent")
    - Titles and abbreviations (e.g., "Mr." → "Mister")
    - Years (e.g., "1999" → "nineteen ninety-nine")
    - Large numbers (e.g., "1000000" → "one million")
    - Decimals (e.g., "0.5" → "zero point five")
    - Mixed text (e.g., "The meeting is at 3:00pm on 1/1/2024.")
    - And more...


    Example usage:
        >>> converter = SpokenTextConverter()
        >>> result = converter.convert_to_spoken_text("The meeting is at 3:00pm on 1/1/2024.")
        >>> print(result)
        The meeting is at three o'clock on one/one/twenty twenty-four.
    """

    def __init__(self):
        # Initialize any necessary state or configurations here, maybe for other languages?
        pass

    def _number_to_words(self, num):
        """
        Convert a number into its spoken-word equivalent.

        This method handles both integers and floating-point numbers, including
        negative numbers and large numbers (e.g., "1000000" → "one million").

        :param num: The number to convert. Can be an integer, float, or string.
        :type num: int, float, or str
        :return: The spoken-word equivalent of the number.
        :rtype: str
        :raises ValueError: If the input is not a valid number.
        """
        try:
            if isinstance(num, str):
                # Check if it's actually an integer in string form
                if "." not in num or num.endswith(".0"):
                    num = int(float(num))
                else:
                    num = float(num)

            # Special handling for integers
            if isinstance(num, int) or (isinstance(num, float) and num.is_integer()):
                num = int(num)  # Convert to int if it's a whole number

            if num == 0:
                return "zero"

            ones = [
                "zero",
                "one",
                "two",
                "three",
                "four",
                "five",
                "six",
                "seven",
                "eight",
                "nine",
                "ten",
                "eleven",
                "twelve",
                "thirteen",
                "fourteen",
                "fifteen",
                "sixteen",
                "seventeen",
                "eighteen",
                "nineteen",
            ]
            tens = [
                "",
                "",
                "twenty",
                "thirty",
                "forty",
                "fifty",
                "sixty",
                "seventy",
                "eighty",
                "ninety",
            ]
            scales = ["", "thousand", "million", "billion"]

            def process_chunk(n, scale):
                if n == 0:
                    return ""

                hundreds = n // 100
                remainder = n % 100

                words = []

                if hundreds > 0:
                    words.append(f"{ones[hundreds]} hundred")

                if remainder > 0:
                    if remainder < 20:
                        words.append(ones[remainder])
                    else:
                        tens_digit = remainder // 10
                        ones_digit = remainder % 10
                        if ones_digit == 0:
                            words.append(tens[tens_digit])
                        else:
                            words.append(f"{tens[tens_digit]}-{ones[ones_digit]}")

                if scale > 0 and len(words) > 0:
                    words.append(scales[scale])

                return " ".join(words)

            # Handle negative numbers
            if num < 0:
                return "negative " + self._number_to_words(abs(num))

            # Handle whole numbers differently from decimals
            if isinstance(num, int):
                if num == 0:
                    return "zero"

                result = []
                scale = 0

                while num > 0:
                    chunk = num % 1000
                    if chunk != 0:
                        chunk_words = process_chunk(chunk, scale)
                        result.insert(0, chunk_words)
                    num //= 1000
                    scale += 1

                return " ".join(filter(None, result))
            else:
                # Handle decimal numbers
                str_num = f"{num:.10f}".rstrip("0")  # Handle floating point precision
                if "." in str_num:
                    int_part, dec_part = str_num.split(".")
                else:
                    int_part, dec_part = str_num, ""

                int_num = int(int_part)

                # Convert integer part
                if int_num == 0:
                    result = "zero"
                else:
                    result = []
                    scale = 0
                    while int_num > 0:
                        chunk = int_num % 1000
                        if chunk != 0:
                            chunk_words = process_chunk(chunk, scale)
                            result.insert(0, chunk_words)
                        int_num //= 1000
                        scale += 1
                    result = " ".join(filter(None, result))

                # Add decimal part if it exists
                if dec_part:
                    return (
                        result
                        + " point "
                        + " ".join(ones[int(digit)] for digit in dec_part)
                    )
                return result
        except (ValueError, TypeError) as e:
            raise ValueError(f"Invalid number format: {num}") from e

    def _split_num(self, num):
        """
        Convert numbers, times, and years into their spoken-word equivalents.

        This method handles:
        - Times (e.g., "3:00" → "three o'clock")
        - Years (e.g., "1999" → "nineteen ninety-nine")
        - Regular numbers (e.g., "1000" → "one thousand")

        :param num: The input number, time, or year to convert.
        :type num: str
        :return: The spoken-word equivalent of the input.
        :rtype: str
        """
        try:
            num = num.group()
            if "." in num:
                return self._number_to_words(float(num))
            elif ":" in num:
                try:
                    h, m = [int(n) for n in num.split(":")]
                    if not (0 <= h <= 23 and 0 <= m <= 59):
                        return num  # Return original if invalid time
                    if m == 0:
                        return f"{self._number_to_words(h)} o'clock"
                    elif m < 10:
                        return (
                            f"{self._number_to_words(h)} oh {self._number_to_words(m)}"
                        )
                    return f"{self._number_to_words(h)} {self._number_to_words(m)}"
                except ValueError:
                    return num

            # Year handling
            try:
                year = int(num[:4])
                if len(num) == 4 or (len(num) == 5 and num.endswith("s")):
                    if 1000 <= year <= 9999:
                        left, right = divmod(year, 100)
                        s = "s" if num.endswith("s") else ""

                        # Special case for 2000 and 2000s
                        if year == 2000:
                            if s:
                                return "twenty hundreds"  # Handle "2000s"
                            else:
                                return "two thousand"  # Handle "2000"
                        elif right == 0:
                            return f"{self._number_to_words(left)} hundred{s}"
                        elif right < 10:
                            return f"{self._number_to_words(left)} oh {self._number_to_words(right)}{s}"
                        else:
                            # Handle plural for decades (e.g., 1950s → "nineteen fifties")
                            if s and right >= 10:
                                decade_word = self._number_to_words(right).replace(
                                    " ", "-"
                                )
                                if decade_word.endswith("y"):
                                    decade_word = (
                                        decade_word[:-1] + "ies"
                                    )  # Replace "y" with "ies"
                                else:
                                    decade_word += "s"
                                return f"{self._number_to_words(left)} {decade_word}"
                            return f"{self._number_to_words(left)} {self._number_to_words(right)}{s}"
                return num
            except ValueError:
                return num
        except Exception:
            return num

    def _flip_money(self, m):
        """
        Convert currency expressions into their spoken-word equivalents.

        This method handles both dollars and pounds, including cents and pence.

        :param m: The input currency expression (e.g., "$50.00").
        :type m: str
        :return: The spoken-word equivalent of the currency expression.
        :rtype: str
        :raises ValueError: If the currency format is invalid.
        """
        try:
            m = m.group()
            if not m or len(m) < 2:
                raise ValueError("Invalid currency format")

            bill = "dollar" if m[0] == "$" else "pound"
            amount_str = m[1:]

            if amount_str.isalpha():
                return f"{self._number_to_words(int(amount_str))} {bill}s"
            elif "." not in amount_str:
                amount = int(amount_str)
                s = "" if amount == 1 else "s"
                return f"{self._number_to_words(amount)} {bill}{s}"

            try:
                b, c = amount_str.split(".")
                if not b:  # Handle case like "$.50"
                    b = "0"
                s = "" if b == "1" else "s"
                c = int(c.ljust(2, "0"))

                # Don't add cents/pence if it's zero
                if c == 0:
                    return f"{self._number_to_words(int(b))} {bill}{s}"

                coins = (
                    f"cent{'' if c == 1 else 's'}"
                    if m[0] == "$"
                    else ("penny" if c == 1 else "pence")
                )
                return f"{self._number_to_words(int(b))} {bill}{s} and {self._number_to_words(c)} {coins}"
            except ValueError as e:
                raise ValueError(f"Invalid currency format: {m}") from e
        except Exception:
            return m  # Return original text if conversion fails

    def _point_num(self, num) -> str:
        """
        Convert decimal numbers into their spoken-word equivalents.

        :param num: The input decimal number (e.g., "3.14").
        :type num: str
        :return: The spoken-word equivalent of the decimal number.
        :rtype: str
        """
        return self._number_to_words(float(num.group()))

    def _convert_percentages(self, text):
        """
        Convert percentage expressions in the text to their spoken-word equivalents.

        :param text: The input text containing percentage expressions (e.g., "50%").
        :type text: str
        :return: The input text with percentages converted to spoken words.
        :rtype: str
        """

        def replace_match(match):
            number = match.group(1)
            # Handle whole numbers without decimal point
            if "." not in number:
                return f"{self._number_to_words(int(number))} percent"
            return f"{self._number_to_words(float(number))} percent"

        return re.sub(r"(\d+\.?\d*)%", replace_match, text)

    def _convert_mathematical_notation(self, text: str) -> str:
        """
        Convert mathematical notation to spoken form.

        Handles:
        - Exponents (e.g., "8^2" → "eight to the power of two")
        - Square/cube roots (e.g., "√9" → "square root of nine")
        - Basic operations (e.g., "5 + 3" → "five plus three")
        - Fractions (e.g., "1/2" → "one over two")
        - Equals signs (e.g., "=" → "equals")
        - Division signs (e.g., "÷" → "divided by")
        - Multiplication signs (e.g., "×" → "times")

        :param text: Text containing mathematical notation
        :return: Text with mathematical notation converted to spoken form
        """

        # Helper function to convert numbers in matched patterns
        def convert_numbers_in_match(match, pattern):
            parts = list(match.groups())
            for i, part in enumerate(parts):
                if part and part.isdigit():
                    parts[i] = self._number_to_words(int(part))
            return pattern.format(*parts)

        # Convert basic arithmetic symbols first
        text = text.replace(" = ", " equals ")
        text = text.replace("=", " equals ")
        text = text.replace(" + ", " plus ")
        text = text.replace("+", " plus ")
        text = text.replace(" - ", " minus ")
        text = text.replace(" × ", " times ")
        text = text.replace("×", " times ")
        text = text.replace(" ÷ ", " divided by ")
        text = text.replace("÷", " divided by ")

        # Convert exponents (e.g., 8^2, x^2, etc.)
        text = re.sub(
            r"(\d+)\^(\d+)",
            lambda m: convert_numbers_in_match(m, "{0} to the power of {1}"),
            text,
        )

        # Convert letter variables with exponents (e.g., x^2)
        text = re.sub(
            r"([a-zA-Z])\^(\d+)",
            lambda m: f"{m.group(1)} to the power of {self._number_to_words(int(m.group(2)))}",
            text,
        )

        # Convert square roots (√)
        text = re.sub(
            r"√(\d+)",
            lambda m: f"square root of {self._number_to_words(int(m.group(1)))}",
            text,
        )

        # Convert cube roots (∛)
        text = re.sub(
            r"∛(\d+)",
            lambda m: f"cube root of {self._number_to_words(int(m.group(1)))}",
            text,
        )

        # Convert mathematical fractions (only if not part of a date)
        def convert_fraction(match):
            # Skip if it looks like a date (e.g., 1/1/2024)
            if re.match(r"\d{1,2}/\d{1,2}/\d{2,4}", match.group(0)):
                return match.group(0)
            num = self._number_to_words(int(match.group(1)))
            den = self._number_to_words(int(match.group(2)))
            return f"{num} over {den}"

        text = re.sub(r"(\d+)/(\d+)(?!/)", convert_fraction, text)

        # Clean up any extra spaces that may have been introduced
        text = re.sub(r"\s+", " ", text).strip()

        return text

    def text_to_spoken(self, text) -> str:
        """
        Convert a given text into its spoken-word equivalent.

        This method processes the input text, identifying and converting numbers, dates,
        times, currency, and percentages into their spoken-word forms.

        :param text: The input text to convert.
        :type text: str
        :return: The input text with numbers, dates, times, and currency converted to spoken words.
        :rtype: str
        :raises ValueError: If the input text contains invalid or unsupported formats.
        """
        # 1. Basic text cleanup
        # remove leading and trailing whitespace and empty lines
        text = "\n".join(line.strip() for line in text.splitlines() if line.strip())

        # 2. Quote normalization
        text = text.replace(chr(8216), "'").replace(chr(8217), "'")
        text = text.replace("«", chr(8220)).replace("»", chr(8221))
        text = text.replace(chr(8220), '"').replace(chr(8221), '"')
        text = text.replace("(", "«").replace(")", "»")

        # 3. Punctuation normalization
        for a, b in zip("、。！，：；？", ",.!,:;?"):
            text = text.replace(a, b + " ")

        # 4. Whitespace normalization
        text = re.sub(r"[^\S \n]", " ", text)
        text = re.sub(r"  +", " ", text)
        text = re.sub(r"(?<=\n) +(?=\n)", "", text)

        # 5. Convert titles and abbreviations
        text = re.sub(r"\bD[Rr]\.(?= [A-Z])", "Doctor", text)
        text = re.sub(r"\b(?:Mr\.|MR\.(?= [A-Z]))", "Mister", text)
        text = re.sub(r"\b(?:Ms\.|MS\.(?= [A-Z]))", "Miss", text)
        text = re.sub(r"\b(?:Mrs\.|MRS\.(?= [A-Z]))", "Mrs", text)
        text = re.sub(r"\betc\.(?! [A-Z])", "etc", text)
        text = re.sub(r"(?i)\b(y)eah?\b", r"\1e'a", text)

        # 6. Number formatting preparation
        # Remove commas in numbers but preserve them for later conversion
        def preserve_large_numbers(match):
            num = int(match.group().replace(",", ""))
            return self._number_to_words(num)

        text = re.sub(r"\b\d{1,3}(?:,\d{3})+\b", preserve_large_numbers, text)
        text = re.sub(r"(?<=\d),(?=\d)", "", text)

        # 7. Remove AM/PM but preserve the time part
        text = re.sub(r"(\d+:\d+)\s*(?:am|pm)\b", r"\1", text, flags=re.IGNORECASE)

        # 8. Date conversion (before other number conversions)
        def convert_date(match):
            parts = match.group().split("/")
            if len(parts) == 3 and len(parts[2]) == 4:
                # Convert the year part separately
                year = int(parts[2])
                if year == 2000:
                    year_text = "two thousand"
                else:
                    left, right = divmod(year, 100)
                    if right == 0:
                        year_text = f"{self._number_to_words(left)} hundred"
                    else:
                        year_text = f"{self._number_to_words(left)} {self._number_to_words(right)}"
                return f"{self._number_to_words(int(parts[0]))}/{self._number_to_words(int(parts[1]))}/{year_text}"
            return "/".join(self._number_to_words(int(part)) for part in parts)

        text = re.sub(r"\b\d{1,2}/\d{1,2}/(?:\d{4}|\d{2})\b", convert_date, text)

        # 9. Convert mathematical notation (before other number conversions)
        text = self._convert_mathematical_notation(text)

        # 10. Number conversions in specific order:
        # a. Percentages first
        text = self._convert_percentages(text)

        # b. Currency
        text = re.sub(
            r"(?i)[$£]\d+(?:\.\d+)?(?: hundred| thousand| (?:[bm]|tr)illion)*\b|[$£]\d+\.\d\d?\b",
            self._flip_money,
            text,
        )

        # c. Times
        text = re.sub(
            r"\b(?:[1-9]|1[0-2]):[0-5]\d\b",
            self._split_num,
            text,
        )

        # d. Years
        text = re.sub(
            r"\b\d{4}s?\b",
            self._split_num,
            text,
        )

        # e. Decimal numbers
        text = re.sub(r"\d*\.\d+", self._point_num, text)

        # f. Standalone integers (new addition)
        text = re.sub(r"\b\d+\b", lambda m: self._number_to_words(int(m.group())), text)

        # 10. Final formatting
        text = re.sub(r"(?<=\d)-(?=\d)", " to ", text)
        text = re.sub(r"(?<=\d)S", " S", text)
        text = re.sub(r"(?<=[BCDFGHJ-NP-TV-Z])'?s\b", "'S", text)
        text = re.sub(r"(?<=X')S\b", "s", text)
        text = re.sub(
            r"(?:[A-Za-z]\.){2,} [a-z]", lambda m: m.group().replace(".", "-"), text
        )
        text = re.sub(r"(?i)(?<=[A-Z])\.(?=[A-Z])", "-", text)

        # 11. Final cleanup
        text = re.sub(r"\b(?:am|pm)\b", "", text, flags=re.IGNORECASE)
        text = re.sub(
            r"  +", " ", text
        )  # Clean up any double spaces that may have been created

        return text.strip()
