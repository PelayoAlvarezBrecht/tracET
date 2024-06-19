
import sys, os, getopt, time
from src.tracET.core.vtk_uts import *
from src.tracET.core import lio
from src.tracET.representation.graphs import *
from src.tracET.representation.curve import *
#import nrrd
import pandas as pd

def main(argv):
    start = time.time()

    #Input parsing
    in_tomo = None
    r, s = None, None
    t,b=None,None
    try:
        opts, args = getopt.getopt(argv, "hi:r:s:t:b:",["help","itomo","rad","subsam","type","branch"])
    except getopt.GetoptError:
        print('python trace_graph.py -i <in_tomo> -r <radius> -s <subsampling> -o <out_dir>')
        sys.exit()
    for opt, arg in opts:
        if opt in ("-h","--help"):
            print('python trace_graph.py - m <main_dir> -i <in_tomo> -r <radius> -s <subsampling> -o <out_dir>')
            print('\t-i (--itomo) <in_tomo> input tomogram (point cloud)')
            print('\t-r (--rad) <radius> radius to connect points in the graph')
            print('\t-s (--subsam) <subsampling> radius of subsampling (optional, default no subsampling)')
            print('\t-t (--type) <type of filament> "l" (linear) or "n" (net) (optional, default net)')
            print('\t-b (--branch) <branch grade> times we repeat the branch removal for branches with more than one edge. (optional, default 1. Only for linear)')
        elif opt in ("-i","--itomo"):
            in_tomo=arg
            if not(os.path.splitext(in_tomo)[1] in ('.mrc', '.nhdr', '.nrrd')):
                print('The input file must have a .mrc, .nhdr or nrrd extension!')
                sys.exit()
        elif opt in ("-r","--rad"):
            r=arg
        elif opt in ("-s","--subsam"):
            s=arg
        elif opt in ("-t","--type"):
            t=arg
        elif opt in ("-b","--branch"):
            b=arg

    if in_tomo is not None:
        print('\t-Loading input tomogram:', in_tomo)
        if os.path.splitext(in_tomo)[1] == '.mrc':
            T = lio.load_mrc(in_tomo)
        else:
            T = nrrd.read(in_tomo)[0]
    else:
        print('python trace_graph.py -i <in_tomo> -r <radius> -s <subsampling> -o <out_dir>')
        print('Almost an input tomogram -i must be provided')
        sys.exit()
    if r is not None:
        print('Radius = ',str(r))
    else:
        print('python trace_graph.py -i <in_tomo> -r <radius> -s <subsampling> -o <out_dir>')
        print('Almost an input radius -r must be provided')
        sys.exit()
    if s is not None:
        print ('Subsampling radius = ',str(s))
    else:
        s=0
        print('Default no subsampling')
    if t is not None:
        print('Type of filament = ',str(t))
    else:
        t='n'
        print('Default: net filament')
    if b is not None:
        print('Branch grade = ',str(b))
    else:
        b=1
        print('Default: branches grade one')


    print('Calculating the graph')
    [coords, graph_array] =make_skeleton_graph(T,float(r),float(s))

    print('Spliting in components')
    [graph_ar_comps,coords_comps]=split_into_components(graph_array,coords)
    print('Making graphs line like and save')
    tubule_list=np.zeros(len(coords))
    branch_list=np.zeros(len(coords))
    b=0
    a=0
    out_poly = vtk.vtkPolyData()
    #append_comps = vtk.vtkAppendPolyData()
    for i in range(len(graph_ar_comps)):
        print('Procesing tubule ',str(i))

        print('Removing cycles')
        L_graph=spannig_tree_apply(graph_ar_comps[i])

        if t=='l':
            print('For a linear filament, we remove the shortest branches')
            L_coords = coords_comps[i]
            for number in range(int(b)):
                L_graph,L_coords=remove_branches(L_graph,L_coords)
                L_branches=np.ones((len(L_coords)))
                print(number)
        else:
            print('For a net, we leave the branches')
            L_graph,L_coords,L_branches=label_branches2(L_graph,coords_comps[i])


        ##curve processign
        graphs_branch,coords_branchs = split_into_components(L_graph,L_coords)
        for j in range(len(graphs_branch)):
            print('Processing branch '+str(j))
            if len(coords_branchs[j])>1:
                sorted_coords=sort_branches(graphs_branch[j],coords_branchs[j])
                curve = SpaceCurve(sorted_coords)
                curve_poly = curve.get_vtp()
                add_label_to_poly(curve_poly, i, 'Component')
                add_label_to_poly(curve_poly, j, 'Branch')

                out_poly = merge_polys(out_poly, curve_poly)
            try:
                branch_list[b:b + len(coords_branchs[j])] = j * np.ones(len(coords_branchs[j]))

                b = b + len(coords_branchs[j])
            except ValueError:
                continue




        tubule_list[a:a+len(coords_comps[i])]=i*np.ones(len(coords_comps[i]))
        a=a+len(coords_comps[i])
    out_mat=np.zeros((len(coords),5))
    out_mat[:,0]=tubule_list
    out_mat[:,1]=branch_list
    out_mat[:,2:5]=coords
    out_pd=pd.DataFrame(data=out_mat,columns=['Component','Branch','X','Y','Z'])
    out_pd.to_csv(os.path.splitext(in_tomo)[0]+ '_skel_graph.csv')

    save_vtp(out_poly, os.path.splitext(in_tomo)[0] + '_skel_graph.vtp')
    end = time.time()
    print('The program lasted ', str(end - start), ' s in execute')
    print('Successfully terminated. (' + time.strftime("%c") + ')')

if __name__ == "__main__":
    main(sys.argv[1:])