from experiments import setup
setup.run()

from experiments import exp_pytorch as ex

if(__name__ == '__main__'):
    args = ex.getParser().parse_args()
    ex.main(args=args)