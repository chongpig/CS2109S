from prepare_data import * # loads the `get_...` helper funtions

df = get_data()
cases_cumulative = get_n_cases_cumulative(df)
deaths_cumulative = get_n_deaths_cumulative(df)
healthcare_spending = get_healthcare_spending(df)
mask_prices = get_mask_prices(healthcare_spending.shape[1])
stringency_values = get_stringency_values(df)
cases_top_cumulative = get_n_cases_top_cumulative(df)

def compute_stringency_index(stringency_values):
    '''
    Computes the daily stringency index for each country.
    Parameters
    ----------
    stringency_values: np.ndarray
        3D `ndarray` with each row representing the data of a country, and the columns
        of each row representing the time series data of the stringency values as a
        vector. To be specific, on each day, there are four different stringency
        values for 'school closing', 'workplace closing', 'stay at home requirements'
        and 'international travel controls', respectively. For instance, the (i, j, 0)
        entry represents the `school closing` stringency value for the ith country
        on the (j + 1)th day.
    
    Returns
    -------
    Daily stringency index for each country as a 2D `ndarray` such that the (i, j)
    entry corresponds to the stringency index in the ith country on the (j + 1)th
    day.
    In this case, we shall assume that 'stay at home requirements' is the most
    restrictive regulation among the other regulations, 'international travel
    controls' is more restrictive than 'school closing' and 'workplace closing',
    and 'school closing' and 'workplace closing' are equally restrictive. Thus,
    to compute the stringency index, we shall weigh each stringency value by 1,
    1, 3 and 2 for 'school closing', 'workplace closing', 'stay at home
    requirements' and 'international travel controls', respectively. Then, the 
    index for the ith country on the (j + 1)th day is given by
    `stringency_values[i, j, 0] + stringency_values[i, j, 1] +
    3 * stringency_values[i, j, 2] + 2 * stringency_values[i, j, 3]`.
    Note
    ----
    Use matrix operations and broadcasting to complete this question. Please do
    not use iterative approaches like for-loops.
    '''
    """ YOUR CODE HERE """
    weights = np.array([1, 1, 3, 2]).reshape(1, 4, 1)
    return (stringency_values @ weights)[:, :, 0]
    """ YOUR CODE END HERE """

# Test cases
stringency_values = np.ones((10, 20, 4))
stringency_values[:, 10:, :] *= 2
actual = compute_stringency_index(stringency_values)
expected = np.ones((10, 20)) * (1 + 1 + 3 + 2)
expected[:, 10:] *= 2
assert(np.all(actual == expected))

stringency_values2 = np.array([[[0, 0, 0, 0], [1, 0, 0, 0]], [[0, 0, 0, 0], [0, 1, 2, 0]]])
actual2 = compute_stringency_index(stringency_values2)
expected2 = np.array([[0, 1], [0, 7]])
assert(np.all(actual2 == expected2))
