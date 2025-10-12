from prepare_data import * # loads the `get_...` helper funtions

df = get_data()
cases_cumulative = get_n_cases_cumulative(df)
deaths_cumulative = get_n_deaths_cumulative(df)
healthcare_spending = get_healthcare_spending(df)
mask_prices = get_mask_prices(healthcare_spending.shape[1])
stringency_values = get_stringency_values(df)
cases_top_cumulative = get_n_cases_top_cumulative(df)

def is_peak(n_cases_increase_avg, n_adj_entries_peak=7):
    '''
    Determines whether the (j + 1)th day was a day when the increase in cases
    peaked in the ith country.
    Parameters
    ----------
    n_cases_increase_avg: np.ndarray
        2D `ndarray` with each row representing the data of a country, and the columns
        of each row representing the time series data of the average daily increase in the
        number of confirmed cases in that country, i.e. the ith row of 
        `n_cases_increase` contains the data of the ith country, and the (i, j) entry of
        `n_cases_increase` is the average daily increase in the number of confirmed
        cases on the (j + 1)th day in the ith country. In this case, the 'average'
        is computed using the output from `average_increase_in_cases`.
    n_adj_entries_peak: int
        Number of days that determines the size of the window in which peaks are
        to be detected. 
    
    Returns
    -------
    2D `ndarray` with the (i, j) entry indicating whether there is a peak in the
    daily increase in cases on the (j + 1)th day in the ith country.
    Suppose `a` is the average daily increase in cases, with the (i, j) entry
    indicating the average increase in cases on the (j + 1)th day in the ith
    country. Moreover, let `n_adj_entries_peak` be denoted by `m`.
    In addition, an increase on the (j + 1)th day is deemed significant in the
    ith country if `a[i, j]` is greater than 10 percent of the mean of all
    average daily increases in the country.
    Now, to determine whether there is a peak on the (j + 1)th day in the ith
    country, check whether `a[i, j]` is maximum in {`a[i, j - m]`, `a[i, j - m + 1]`,
    ..., `a[i, j + m - 1]`, `a[i, j + m]`}. If it is and `a[i, j]` is significant,
    then there is a peak on the (j + 1)th day in the ith country; otherwise,
    there is no peak.
    Note
    ----
    Let d = `n_adj_entries_avg` + `n_adj_entries_peak`, where `n_adj_entries_avg`
    is that used to compute `n_cases_increase_avg`. Observe that it is not
    possible to detect a peak in the first and last d days, i.e. these days should
    not be peaks.
    
    As described in `average_increase_in_cases`, to compute the average daily
    increase, we need data from the previous and the next `n_adj_entries_avg`
    number of days. Hence, we won't have an average for these days, precluding
    the computation of peaks during the first and last `n_adj_entries_avg` days.
    Moreover, similar to `average_increase_in_cases`, we need the data over the
    interval [-`n_adj_entries_peak` + j, `n_adj_entries_peak` + j] to determine
    whether the (j + 1)th day is a peak.
    Hint: to determine `n_adj_entries_avg` from `n_cases_increase_avg`,
    `np.count_nonzero` and `np.isnan` may be helpful.

    Your implementation should not involve any iteration, including `map` and `filter`, 
    recursion, or any iterative approaches like for-loops.
    '''
    """ YOUR CODE HERE """
    countries, days = n_cases_increase_avg.shape
    m = n_adj_entries_peak
    numNans = np.isnan(n_cases_increase_avg[0,:]).sum() // 2
    notPeak = numNans + m
    result = np.zeros((countries, days), dtype = bool)
    countriesMean = np.nanmean(n_cases_increase_avg, axis = 1, keepdims = True)
    significant = n_cases_increase_avg > 0.1 * countriesMean
    windows = np.lib.stride_tricks.sliding_window_view(n_cases_increase_avg,
                                                        2 * m + 1,
                                                        axis = 1)
    maximums = np.nanmax(windows, axis = 2)
    isMax = windows[:,:,m] == maximums
    firstMaxs = np.argmax(windows, axis = 2)
    isTrueMax = firstMaxs >= m
    isPeak = isMax & isTrueMax
    result[:,m:days-m] = isPeak & significant[:,m:days-m]
    result[:, :notPeak] = False
    result[:, -notPeak:] = False
    return result
    
    """ YOUR CODE END HERE """

    # Test cases
n_cases_increase_avg = np.array([[np.nan, np.nan, 10, 10, 5, 20, 7, np.nan, np.nan], [np.nan, np.nan, 15, 5, 16, 17, 17, np.nan, np.nan]])
n_adj_entries_peak = 1

actual = is_peak(n_cases_increase_avg, n_adj_entries_peak=n_adj_entries_peak)
expected = np.array([[False, False, False, False, False, True, False, False, False],
                     [False, False, False, False, False, True, False, False, False]])
assert np.all(actual == expected)

n_cases_increase_avg2 = np.array([[np.nan, np.nan, 10, 20, 20, 20, 20, np.nan, np.nan], [np.nan, np.nan, 20, 20, 20, 20, 10, np.nan, np.nan]])
n_adj_entries_peak2 = 1

actual2 = is_peak(n_cases_increase_avg2, n_adj_entries_peak=n_adj_entries_peak2)
expected2 = np.array([[False, False, False, True, False, False, False, False, False],
                    [False, False, False, False, False, False, False, False, False]])
assert np.all(actual2 == expected2)
