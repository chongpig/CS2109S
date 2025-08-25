from prepare_data import * # loads the `get_...` helper funtions

df = get_data()
cases_cumulative = get_n_cases_cumulative(df)
deaths_cumulative = get_n_deaths_cumulative(df)
healthcare_spending = get_healthcare_spending(df)
mask_prices = get_mask_prices(healthcare_spending.shape[1])
stringency_values = get_stringency_values(df)
cases_top_cumulative = get_n_cases_top_cumulative(df)

def average_increase_in_cases(n_cases_increase, n_adj_entries_avg=7):
    '''
    Averages the increase in cases for each day using data from the previous
    `n_adj_entries_avg` number of days and the next `n_adj_entries_avg` number
    of days.
    Parameters
    ----------
    n_cases_increase: np.ndarray
        2D `ndarray` with each row representing the data of a country, and the columns
        of each row representing the time series data of the daily increase in the
        number of confirmed cases in that country, i.e. the ith row of 
        `n_cases_increase` contains the data of the ith country, and the (i, j) entry of
        `n_cases_increase` is the daily increase in the number of confirmed cases on the
        (j + 1)th day in the ith country.
    n_adj_entries_avg: int
        Number of days from which data will be used to compute the average increase
        in cases. This should be a positive integer.
    
    Returns
    -------
    Mean increase in cases for each day, using data from the previous
    `n_adj_entries_avg` number of days and the next `n_adj_entries_avg` number
    of days, as a 2D `ndarray` such that the (i, j) entry represents the
    average increase in daily cases on the (j + 1)th day in the ith country,
    rounded down to the smallest integer.
    
    The average increase in cases for a particular country on the (j + 1)th day
    is given by the mean of the daily increase in cases over the interval
    [-`n_adj_entries_avg` + j, `n_adj_entries_avg` + j]. (Note: this interval
    includes the endpoints).
    Note
    ----
    Since this computation requires data from the previous `n_adj_entries_avg`
    number of days and the next `n_adj_entries_avg` number of days, it is not
    possible to compute the average for the first and last `n_adj_entries_avg`
    number of days. Therefore, set the average increase in cases for these days
    to `np.nan` for all countries.
    Your implementation should not involve any iteration, including `map` and `filter`, 
    recursion, or any iterative approaches like for-loops.
    '''
    """ YOUR CODE HERE """
    countries = n_cases_increase.shape[0]
    days = n_cases_increase.shape[1]
    windowWidth = 2 * n_adj_entries_avg + 1
    cutCases = np.lib.stride_tricks.sliding_window_view(n_cases_increase, windowWidth,axis = 1)
    means = np.floor(np.mean(cutCases,axis = 2))
    result = np.full((countries, days), np.nan)
    result[:, n_adj_entries_avg:days - n_adj_entries_avg] = means
    return result
    """ YOUR CODE END HERE """

# Test cases
n_cases_increase = np.array([[0, 5, 10, 15, 20, 25, 30]])
actual = average_increase_in_cases(n_cases_increase, n_adj_entries_avg=2)
expected = np.array([[np.nan, np.nan, 10, 15, 20, np.nan, np.nan]])
assert(np.array_equal(actual, expected, equal_nan=True))
