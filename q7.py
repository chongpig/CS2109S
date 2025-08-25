from prepare_data import * # loads the `get_...` helper funtions

df = get_data()
cases_cumulative = get_n_cases_cumulative(df)
deaths_cumulative = get_n_deaths_cumulative(df)
healthcare_spending = get_healthcare_spending(df)
mask_prices = get_mask_prices(healthcare_spending.shape[1])
stringency_values = get_stringency_values(df)
cases_top_cumulative = get_n_cases_top_cumulative(df)

def find_max_increase_in_cases(n_cases_increase):
    '''
    Finds the maximum daily increase in confirmed cases for each country.
    Parameters
    ----------
    n_cases_increase: np.ndarray
        2D `ndarray` with each row representing the data of a country, and the columns
        of each row representing the time series data of the daily increase in the
        number of confirmed cases in that country, i.e. the ith row of 
        `n_cases_increase` contains the data of the ith country, and the (i, j) entry of
        `n_cases_increase` is the daily increase in the number of confirmed cases on the
        (j + 1)th day in the ith country.
    
    Returns
    -------
    Maximum daily increase in cases for each country as a 1D `ndarray` such that the
    ith entry corresponds to the increase in confirmed cases in the ith country as
    represented in `n_cases_increase`.
    Your implementation should not involve any iteration, including `map` and `filter`, 
    recursion, or any iterative approaches like for-loops.
    '''
    """ YOUR CODE HERE """
    return np.amax(n_cases_increase, axis=1)
    """ YOUR CODE END HERE """

# Test cases
n_cases_increase = np.ones((100, 20))
actual = find_max_increase_in_cases(n_cases_increase)
expected = np.ones(100)
assert(np.all(actual == expected))

sample_increase = np.array([[1,2,3,4,8,8,10,10,10,10],[1,1,3,5,8,10,15,20,25,30]])
expected2 = np.array([10, 30]) # max of [1,2,3,4,8,8,10,10,10,10] => 10, max of [1,1,3,5,8,10,15,20,25,30] => 30
assert(np.all(find_max_increase_in_cases(sample_increase) == expected2))

sample_increase2 = np.array([\
            [51764, 84, 159, 140, 183, 0],\
            [55755, 499, 318, 574, 581, 589],\
            [97857, 392, 382, 357, 323, 299]])
expected3 = np.array([51764, 55755, 97857])
assert(np.all(find_max_increase_in_cases(sample_increase2) == expected3))

n_cases_increase2 = compute_increase_in_cases(cases_top_cumulative.shape[1], cases_top_cumulative)
expected4 = np.array([ 68699.,  97894., 258110.])
assert(np.all(find_max_increase_in_cases(n_cases_increase2) == expected4))
