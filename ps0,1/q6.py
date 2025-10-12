from prepare_data import * # loads the `get_...` helper funtions

df = get_data()
cases_cumulative = get_n_cases_cumulative(df)
deaths_cumulative = get_n_deaths_cumulative(df)
healthcare_spending = get_healthcare_spending(df)
mask_prices = get_mask_prices(healthcare_spending.shape[1])
stringency_values = get_stringency_values(df)
cases_top_cumulative = get_n_cases_top_cumulative(df)

def compute_increase_in_cases(n, cases_cumulative):
    '''
    Computes the daily increase in confirmed cases for each country for the first n days, starting
    from the first day.
    Parameters
    ----------    
    n: int
        How many days of data to return in the final array. If the input data has fewer
        than n days of data then we just return whatever we have for each country up to n. 
    cases_cumulative: np.ndarray
        2D `ndarray` with each row representing the data of a country, and the columns
        of each row representing the time series data of the cumulative number of
        confirmed cases in that country, i.e. the ith row of `cases_cumulative`
        contains the data of the ith country, and the (i, j) entry of
        `cases_cumulative` is the cumulative number of confirmed cases on the
        (j + 1)th day in the ith country.
    
    Returns
    -------
    Daily increase in cases for each country as a 2D `ndarray` such that the (i, j)
    entry corresponds to the increase in confirmed cases in the ith country on
    the (j + 1)th day, where j is non-negative.
    Note
    ----
    The number of cases on the zeroth day is assumed to be 0, and we want to
    compute the daily increase in cases starting from the first day.
    Your implementation should not involve any iteration, including `map` and `filter`, 
    recursion, or any iterative approaches like for-loops.
    '''
    """ YOUR CODE HERE """
    length = cases_cumulative.shape[1]
    days = min(n,length)
    firstDay  = cases_cumulative[:,0:1]
    otherDay = np.diff(cases_cumulative[:, :n], axis=1)
    result = np.concatenate([firstDay, otherDay], axis=1)
    return result
    """ YOUR CODE END HERE """

# Test cases
cases_cumulative = np.zeros((100, 20))
cases_cumulative[:, :] = np.arange(1, 21)
actual = compute_increase_in_cases(100, cases_cumulative)
assert(np.all(actual == np.ones((100, 20))))

sample_cumulative = np.array([[1,2,3,4,8,8,10,10,10,10],[1,1,3,5,8,10,15,20,25,30]])
expected = np.array([[1, 1, 1, 1, 4.], [1, 0, 2, 2, 3]])
assert(np.all(compute_increase_in_cases(5,sample_cumulative) == expected))

expected2 = np.array([[1, 1, 1, 1, 4, 0, 2, 0, 0, 0],[1, 0, 2, 2, 3, 2, 5, 5, 5, 5]])
assert(np.all(compute_increase_in_cases(10,sample_cumulative) == expected2))
assert(np.all(compute_increase_in_cases(20,sample_cumulative) == expected2))

sample_cumulative2 = np.array([[51764, 51848, 52007, 52147, 52330, 52330],\
                            [55755, 56254, 56572, 57146, 57727, 58316],\
                            [97857, 98249, 98631, 98988, 99311, 99610]])
expected3 = np.array([\
            [51764, 84, 159, 140, 183, 0],\
            [55755, 499, 318, 574, 581, 589],\
            [97857, 392, 382, 357, 323, 299]])
assert(np.all(compute_increase_in_cases(6,sample_cumulative2) == expected3))
