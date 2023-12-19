from pmlb import fetch_data


names = [
        '215_2dplanes',
        '344_mv',
        '562_cpu_small',
        '294_satellite_image',
        '573_cpu_act',
        '227_cpu_small',
        '564_fried',
        '201_pol'
    ]


for name in names:
    X, y = fetch_data(name, return_X_y=True)
    print(name, X.shape, y.shape)
