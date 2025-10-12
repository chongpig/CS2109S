from prepare_data import * # loads the `get_...` helper funtions

df = get_data()
cases_cumulative = get_n_cases_cumulative(df)
deaths_cumulative = get_n_deaths_cumulative(df)
healthcare_spending = get_healthcare_spending(df)
mask_prices = get_mask_prices(healthcare_spending.shape[1])
stringency_values = get_stringency_values(df)
cases_top_cumulative = get_n_cases_top_cumulative(df)

def compute_death_rate_first_n_days(n, cases_cumulative, deaths_cumulative):
    '''
    Computes the average number of deaths recorded for every confirmed case
    that is recorded from the first day to the nth day (inclusive).
    Parameters
    ----------
    n: int
        How many days of data to return in the final array.
    cases_cumulative: np.ndarray
        2D `ndarray` with each row representing the data of a country, and the columns
        of each row representing the time series data of the cumulative number of
        confirmed cases in that country, i.e. the ith row of `cases_cumulative`
        contains the data of the ith country, and the (i, j) entry of
        `cases_cumulative` is the cumulative number of confirmed cases on the
        (j + 1)th day in the ith country.
    deaths_cumulative: np.ndarray
        2D `ndarray` with each row representing the data of a country, and the columns
        of each row representing the time series data of the cumulative number of
        confirmed deaths (as a result of COVID-19) in that country, i.e. the ith
        row of `n_deaths_cumulative` contains the data of the ith country, and
        the (i, j) entry of `n_deaths_cumulative` is the cumulative number of
        confirmed deaths on the (j + 1)th day in the ith country.
    
    Returns
    -------
    Average number of deaths recorded for every confirmed case from the first day
    to the nth day (inclusive) for each country as a 1D `ndarray` such that the
    entry in the ith row corresponds to the death rate in the ith country as
    represented in `cases_cumulative` and `deaths_cumulative`.
    Note
    ----
    `cases_cumulative` and `deaths_cumulative` are such that the ith row in the 
    former and that in the latter contain data of the same country. In addition,
    if there are no confirmed cases for a particular country, the expected death
    rate for that country should be zero. (Hint: to deal with NaN look at
    `np.nan_to_num`)
    Your implementation should not involve any iteration, including `map` and `filter`, 
    recursion, or any iterative approaches like for-loops.
    '''
    """ YOUR CODE HERE """
    newCase = cases_cumulative[:, n-1]
    newDeath = deaths_cumulative[:, n-1]
    ratio = np.nan_to_num(newDeath / newCase, nan=0.0)
    return ratio
    """ YOUR CODE END HERE """

# Test cases
n_cases_cumulative = cases_cumulative[:3, :] #Using data from CSV. Make sure to run relevant cell above
n_deaths_cumulative = deaths_cumulative[:3, :]
expected = np.array([0.0337837838, 0.0562347188, 0.1410564226])
np.testing.assert_allclose(compute_death_rate_first_n_days(100, n_cases_cumulative, n_deaths_cumulative), expected)

sample_cumulative = np.array([[1,2,3,4,8,8,10,10,10,10], [1,2,3,4,8,8,10,10,10,10]])
sample_death = np.array([[0,0,0,1,2,2,2,2,5,5], [0,0,0,1,2,2,2,2,5,5]])

expected2 = np.array([0.5, 0.5])
assert(np.all(compute_death_rate_first_n_days(10, sample_cumulative, sample_death) == expected2))

sample_cumulative2 = np.array([[1,2,3,4,8,8,10,10,10,10]])
sample_death2 = np.array([[0,0,0,1,2,2,2,2,5,5]])

expected3 = np.array([0.5])
assert(compute_death_rate_first_n_days(10, sample_cumulative2, sample_death2) == expected3)
expected4 = np.array([0.25])
assert(compute_death_rate_first_n_days(5, sample_cumulative2, sample_death2) == expected4)
