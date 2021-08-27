from framework import smoothingFramework as sf
import argparse

"""
    Plik służy do rysowania wykresów na podstawie plików csv. 
    Pliki te muszą zawierać jedynie kolejne dane oddzielone od siebie nową linią.
"""

def getParser():
    parser = argparse.ArgumentParser(description='PyTorch Classification Training', add_help=True)
    parser.add_argument('--xl', default='x', help='x label name')
    parser.add_argument('--yl', default='y', help='y label name')
    parser.add_argument('--to', default='.', help='folder where to save plots and files')
    parser.add_argument('--oname', default='weightsSumTrain', help='output file name')
    parser.add_argument('--dpi', type=int, default=300, help='dpi of the plot')
    parser.add_argument('--inchres', type=float, default=6.5, help='resolution in inches')
    parser.add_argument('--widthfreq', type=float, default=0.15, help='frequency of the OX values in the range of (0; 1)')
    parser.add_argument('--pnames', type=str, nargs='+', default=[],
        help='names of the plots')
    parser.add_argument('--paths', type=str, nargs='+', default=[],
        help='paths to the csv files')
    parser.add_argument('--aspectratio', type=float, default=0.3, help='real aspect ratio in inches height / width. Must be > 0')
    parser.add_argument('--fontsize', type=int, default=13, help='font size')
    
    parser.add_argument('--startat', type=int, default=None, help='where to start plot')
    parser.add_argument('--endat', type=int, default=None, help='where to end plot')
    parser.add_argument('--highat', type=float, default=None, help='upper limit of the plot')
    parser.add_argument('--lowat', type=float, default=None, help='lower limit of the plot')

    parser.add_argument('--startscale', type=float, default=None, help='xleft scale of the plot')
    parser.add_argument('--endscale', type=float, default=None, help='xright scale of the plot')
    parser.add_argument('--highscale', type=float, default=None, help='ytop scale of the plot')
    parser.add_argument('--lowscale', type=float, default=None, help='ybottom scale of the plot')
    

    parser.add_argument('--swindow', type=int, default=-1, help='sliding window size. To disable set less than 1.')

    
    return parser

if(__name__ == '__main__'):
    args = getParser().parse_args()

    paths = args.paths
    yl = args.yl
    if(args.endat == -1):
        args.endat = None

    if(args.swindow > 1):
        tmp = sf.Statistics.slidingWindow(fileNamesWithPaths=args.paths, runningAvgSize=args.swindow, outputFolder=args.to, getBaseNameFile=True)
        if(tmp is not None):
            paths = tmp
            yl = yl + ' (okno=' + str(args.swindow) + ')'


    sf.plot(filePath=paths, xlabel=args.xl, ylabel=yl, name=args.oname, plotInputRoot='.', plotsNames=args.pnames,
        plotOutputRoot=args.to, fileFormat=".png", startAt=args.startat, endAt=args.endat, dpi=args.dpi, 
        resolutionInches=args.inchres, widthTickFreq=args.widthfreq, aspectRatio=args.aspectratio, highAt=args.highat, 
        lowAt=args.lowat, fontSize=args.fontsize,
        startScale=args.startscale, endScale=args.endscale, highScale=args.highscale, lowScale=args.lowscale)