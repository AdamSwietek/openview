import multiprocessing as mp

def geoseries_to_mp_dict(row):
    """Convert Pandas objects to picklable dictionary format."""
    return {
        'osmid': row.osmid,
        'geometry': row.geometry.wkt if hasattr(row, 'geometry') else None,  # Convert geometry to WKT string
        'old_geom': row.old_geom.wkt if hasattr(row, 'old_geom') else None,  # Convert geometry to WKT string

        'height': row.height,
        'elevation': row.elevation,
        # 'centroid_x': row.geometry.centroid.x if hasattr(row, 'geometry') else None,
        # 'centroid_y': row.geometry.centroid.y if hasattr(row, 'geometry') else None
    }

def make_structured_array(results):
    import numpy as np

    dtype = [
        ('osmid', np.int64),
        ('vp', np.float64, (3,)),
        ('norm', np.float64, (3,))
    ]
    flattened_results = [item for sublist in results if sublist for item in sublist]
    viewpoints = np.array([(d['osmid'], d['vp'], d['norm']) for d in flattened_results], dtype=dtype)
    return viewpoints

def compute_viewpoints(input_data):
    # Use a relative import so this works when openview is installed as a package.
    from .shape import get_viewpoints_from_footprint

    input_data = [geoseries_to_mp_dict(row) for row in input_data.itertuples()]

    # #get viewpoints using multiprocessing
    # pool = mp.Pool(mp.cpu_count())
    # results = pool.map(get_viewpoints_from_footprint, input_data)
    # pool.close()
    # pool.join()
    from tqdm import tqdm
    import multiprocessing as mp

    with mp.Pool(mp.cpu_count()) as pool:
        results = list(
            tqdm(
                pool.imap_unordered(get_viewpoints_from_footprint, input_data),
                total=len(input_data),
                desc="Computing viewpoints"
            )
        )

    #join list of dictionaries into single dictionary
    viewpoints = make_structured_array(results)
    return viewpoints
