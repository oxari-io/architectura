# Script tries to emulate this command line command
# pyreverse -s1 -A -o dot base/common -d junk -p test --colorized -my
# %%
import pathlib

import pydot
from pylint.pyreverse.main import Run

# for pkgname in ["base.common", "base.dataset_loader"]:
for pkgname in ["base.common"]:
# for pkgname in ["./base/"]:


    FOLDER = pathlib.Path('local/misc')
    print(f"Saving into {FOLDER.absolute()}")

    FOLDER.mkdir(parents=True, exist_ok=True)

    args = [
        "-s1",
        "-A",
        "-odot",
        pkgname,
        f"-d{str(FOLDER.absolute())}",
        "--colorized",
        "-my",
    ]
    try:
        Run(args).run()
    except SystemExit:
        print("Done!")

    (pydot_graph,) = pydot.graph_from_dot_file(FOLDER/"classes.dot")

    for node in pydot_graph.get_node_list():
        # print(f"NODE: {node.get_label()}")
        # print(node.obj_dict, file=open('test.txt', "a"))
        label = node.get_label()
        if label and 'optuna.samplers' in label:
            pydot_graph.del_node(node)
        if label and 'pandas.core' in label:
            pydot_graph.del_node(node)
        if label and (('pathlib.' in label)):
            pydot_graph.del_node(node)
        if label and (('logging.' in label)):
            pydot_graph.del_node(node)
        if label and (('datetime.' in label)):
            pydot_graph.del_node(node)
        if label and (('sklearn.' in label)):
            pydot_graph.del_node(node)
        if label and (('scipy.' in label)):
            pydot_graph.del_node(node)
        if label and (('mapie.' in label)):
            pydot_graph.del_node(node)

    for edge in pydot_graph.get_edge_list():    
        # print(f"EDGE: {edge.get_label()}")
        # print(edge.obj_dict, file=open('test.txt', "a"))
        fontcolor = edge.get_fontcolor()
        # print(fontcolor)
        if fontcolor and "green" in fontcolor:
            edge.set_fontcolor('red')
        # print("")
    
    if  pkgname == "./base/":
        pydot_graph.write_png(FOLDER/"architecture.png")
        pydot_graph.write_svg(FOLDER/"architecture.svg")
        pydot_graph.write_dot(FOLDER/"architecture.dot")
    elif pkgname == "base.common":
        pydot_graph.write_png(FOLDER/"architecture_common.png")
        pydot_graph.write_svg(FOLDER/"architecture_common.svg")
        pydot_graph.write_dot(FOLDER/"architecture_common.dot")
    else:
        pydot_graph.write_png(FOLDER/(pkgname.replace("/", "_")+".png"))
# %%
