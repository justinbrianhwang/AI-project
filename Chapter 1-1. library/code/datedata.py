### datedata.py

## Date Data Processing

def p(str):
    print(str, "\n")

# datetime
import datetime
today = datetime.date.today()
p(today)
p(today.weekday())  # Day of the week
p(today + datetime.timedelta(days=100))  # 100 days later
p(today + datetime.timedelta(days=-100))  # 100 days before
p(today + datetime.timedelta(weeks=3))  # 3 weeks later
p(today + datetime.timedelta(hours=45))  # 45 hours later

day1 = datetime.date(2019, 1, 1)
day2 = datetime.date(2024, 7, 27)
p(day2 - day1)  # Date interval

# calendar
import calendar
p(calendar.weekday(2024, 7, 27))  # Day of the week
p(calendar.isleap(2024))  # Leap year check
