from datetime import datetime

def get_datetime_string():
    date_and_time = datetime.now()
    return date_and_time.strftime('%Y-%m-%d %H:%M:%S')