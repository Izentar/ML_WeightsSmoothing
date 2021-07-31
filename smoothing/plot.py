from framework import smoothingFramework as sf
import argparse

def getParser():
    parser = argparse.ArgumentParser(description='PyTorch Classification Training', add_help=True)
    parser.add_argument('--xl', default='x', help='x label name')
    parser.add_argument('--yl', default='y', help='y label name')
    parser.add_argument('--to', default='.', help='folder to save')
    parser.add_argument('--startat', type=int, default=-10, help='where to start plot')
    parser.add_argument('--oname', default='weightsSumTrain', help='output file name')
    parser.add_argument('--dpi', type=int, default=300, help='dpi of the plot')
    parser.add_argument('--pnames', type=str, nargs='+', default=[],
        help='names of the plots')
    parser.add_argument('--paths', type=str, nargs='+', default=[],
        help='paths to the csv files')
    return parser

if(__name__ == '__main__'):
    args = getParser().parse_args()
    sf.plot(filePath=args.paths, xlabel=args.xl, ylabel=args.yl, name=args.oname, plotInputRoot='.', plotsNames=args.pnames,
        plotOutputRoot=args.to, fileFormat=".png", startAt=args.startat, dpi=args.dpi)