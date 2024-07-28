# exception handling

# result = 10/0

try:
    result = 10/0
except ZeroDivisionError:
    print("Division by zero exception occurred!")
finally:
    print("Executed regardless of the exception!")

class Under19Exception(Exception):
    def __str__(self):
        return "Under 19 exception occurred!"

age = 18
if age < 19:
    try:
        raise Under19Exception
    except Under19Exception:
        print("Handled under 19 exception!")
    finally:
        print("Exception handling complete!")
