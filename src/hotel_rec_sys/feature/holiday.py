import holidays

def add_holiday(df):
    # Define holidays in some countries
    ca_holidays = holidays.Canada()
    us_holidays = holidays.UnitedStates()

    uk_holidays = holidays.UnitedKingdom()
    gr_holidays = holidays.Germany()
    
    # check if checkin or checkout date is in holiday of different countries
    df['north_am_ci'] = df['srch_ci'].apply(lambda x: 1 if x in (us_holidays or ca_holidays)  else 0)
    df['north_am_co'] = df['srch_co'].apply(lambda x: 1 if x in (us_holidays or ca_holidays)  else 0)

    df['europe_ci'] = df['srch_ci'].apply(lambda x: 1 if x in (uk_holidays or gr_holidays)  else 0)
    df['europe_co'] = df['srch_co'].apply(lambda x: 1 if x in (uk_holidays or gr_holidays)  else 0)
    
    # remove original columns
    df= df.drop(['date_time'],axis=1)
    df= df.drop(['srch_ci'],axis=1)
    df= df.drop(['srch_co'],axis=1)
    
    return df