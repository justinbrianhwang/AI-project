# Text Processing Library

# Built-in libraries are provided by Python and do not require separate installation.
# External libraries need to be installed separately after installing Python.
# Use the import statement to use a library.

# Print function
def p(str):
    print(str, "\n")

## textwrap
import textwrap

str = "Hello Python"
# Shorten the string (string, width, placeholder)
p(textwrap.shorten(str, width=10, placeholder="..."))

str = str * 10  # Repeat the string 10 times
p(str)
# Convert the string to a list of 11 elements based on whitespace
wrapstr = textwrap.wrap(str, width=11)
p(wrapstr)

# Convert each element of the list to a string with a newline character
p("\n".join(wrapstr))

## re (regular expression)
# Used to search, extract, and replace substrings in a string using a combination
# of pattern strings and flag strings
# Regular expressions are commonly used across all programming languages, so it is essential to learn them!

import re
str = "Hong Gil-dong's phone number is 010-1234-5678"
pattern = re.compile(r"(\d{3})-(\d{4})-(\d{4})")
p(pattern.sub(r"\g<1> \g<2> \g<3>", str))

# Let's create patterns for phone numbers, emails, IP addresses, social security numbers, etc.!

# Regular expression examples

# 1. Search for "apple" in the string
text = "I like apple pie"
result = re.findall(r"apple", text)
p(result)

# 2. Search for one or more digits
text = "My phone number is 010-1234-5678."
result = re.findall(r"\d+", text)
p(result)

# 3. Simple email address pattern search
text = "email address is example@email.com"
result = re.findall(r"\b\w+@\w+\.\w+", text)
p(result)

# 4. Simple phone number pattern search
text = "Call me at 010-1234-5678"
result = re.findall(r"\d{3}-\d{4}-\d{4}", text)
p(result)

# 5. Search for uppercase English letters
text = "Hello Python"
result = re.findall(r"[A-Z]", text)
p(result)

# 6. Remove unnecessary whitespace in a string
text = "Hello     Python     This is Me"
result = re.sub(r"\s+", " ", text)
p(result)

# 7. Search for the start and end of a string
# ^: start, [^]: negation  ex) ^a: starts with 'a', [^a]: not 'a'
text = "Hello World"
result = re.findall(r"^Hello|World$", text)
p(result)

# 8. Search for strings that start with a specific word
# \b : word boundary
text = "Start your journey with a smile. Start early to avoid traffic."
result = re.findall(r"\bStart\b[^.]*\.", text)
p(result)

# 9. Search for URLs in a string
text = "Website URL is http://example.com or https://www.example.com"
result = re.findall(r"https?://[^\s]+", text)
p(result)

# 10. Search for date formats (YYYY-MM-DD)
text = "Today is 2024-07-27 and tomorrow is 2024-07-28"
result = re.findall(r"\d{4}-\d{2}-\d{2}", text)
p(result)
