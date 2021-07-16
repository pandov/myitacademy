
if __name__ == '__main__':

    from argparse import ArgumentParser
    from multiprocessing import Pool
    from .BIOMETRY import generate as generate_BIOMETRY
    from .FER import generate as generate_FER
    from .CASCADE import generate as generate_CASCADE

    parser = ArgumentParser()
    parser.add_argument('--FER', action='store_true')
    parser.add_argument('--BIOMETRY', action='store_true')
    parser.add_argument('--CASCADE', action='store_true')
    parser.add_argument('--num_workers', default=4, type=int)
    parser.add_argument('--num_classes', type=int)
    args = parser.parse_args()

    with Pool(args.num_workers) as pool:
        if args.FER: pool.map(*generate_FER(num_classes=args.num_classes))
        if args.BIOMETRY: pool.map(*generate_BIOMETRY(num_classes=args.num_classes))
        if args.CASCADE: pool.map(*generate_CASCADE(num_classes=args.num_classes))
