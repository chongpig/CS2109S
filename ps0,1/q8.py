from prepare_data import * # loads the `get_...` helper funtions

df = get_data()
cases_cumulative = get_n_cases_cumulative(df)
deaths_cumulative = get_n_deaths_cumulative(df)
healthcare_spending = get_healthcare_spending(df)
mask_prices = get_mask_prices(healthcare_spending.shape[1])
stringency_values = get_stringency_values(df)
cases_top_cumulative = get_n_cases_top_cumulative(df)

def compute_n_masks_purchaseable(healthcare_spending, mask_prices):
    '''
    Computes the total number of masks that each country can purchase if she
    spends all her emergency healthcare spending on masks.
    Parameters
    ----------
    healthcare_spending: np.ndarray
        2D `ndarray` with each row representing the data of a country, and the columns
        of each row representing the time series data of the emergency healthcare
        spending made by that country, i.e. the ith row of `healthcare_spending`
        contains the data of the ith country, and the (i, j) entry of
        `healthcare_spending` is the amount which the ith country spent on healthcare
        on (j + 1)th day.
    mask_prices: np.ndarray
        1D `ndarray` such that the jth entry represents the cost of 100 masks on the
        (j + 1)th day.
    
    Returns
    -------
    Total number of masks which each country can purchase as a 1D `ndarray` such
    that the ith entry corresponds to the total number of masks purchaseable by the
    ith country as represented in `healthcare_spending`.
    Note
    ----
    The masks can only be bought in batches of 100s.
    Your implementation should not involve any iteration, including `map` and `filter`, 
    recursion, or any iterative approaches like for-loops.
    '''
    """ YOUR CODE HERE """
    maskPerDay = np.floor(healthcare_spending / mask_prices) * 100
    return np.sum(maskPerDay, axis = 1)
    """ YOUR CODE END HERE """

# Test cases
prices_constant = np.ones(5)
healthcare_spending_constant = np.ones((7, 5))
actual = compute_n_masks_purchaseable(healthcare_spending_constant, prices_constant)
expected = np.ones(7) * 500
assert(np.all(actual == expected))

healthcare_spending1 = healthcare_spending[:3, :]  #Using data from CSV
expected2 = [3068779300, 378333500, 6208321700]
assert(np.all(compute_n_masks_purchaseable(healthcare_spending1, mask_prices)==expected2))

healthcare_spending2 = np.array([[0, 100, 0], [100, 0, 200]])
mask_prices2 = np.array([4, 3, 20])
expected3 = np.array([3300, 3500])
assert(np.all(compute_n_masks_purchaseable(healthcare_spending2, mask_prices2)==expected3))
