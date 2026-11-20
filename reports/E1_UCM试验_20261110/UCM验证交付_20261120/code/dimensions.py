"""Apply the pixel reviewer's pre-score tank–shadow classification to reporting only."""
def report_dimension(dimension,template):
    return 'relation' if template=='shadow_direction' else dimension
