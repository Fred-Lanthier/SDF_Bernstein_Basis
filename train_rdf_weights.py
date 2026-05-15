import torch
import rospkg
from src.rdf_weights import RDF_Weights

rospack = rospkg.RosPack()
rospack_path = rospack.get_path('vision_processing')

WS_PATH = rospack_path + '/third_party/SDF_Bernstein_Basis/panda_test'
ROBOT_NAME = 'panda_robot'

# Split the links
FORK_LINK = ['fork_tip']
ROBOT_LINKS = [
    'panda_link0',
    'panda_link1',
    'panda_link2',
    'panda_link3',
    'panda_link4',
    'panda_link5',
    'panda_link6',
    'panda_link7',
    'panda_hand',
    'panda_leftfinger',
    'panda_rightfinger',
]

TRAIN_ITERS = 600

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    rdf = RDF_Weights(device=device)
    rdf.init_robot_folder(WS_PATH, robot_name=ROBOT_NAME)
    
    # 1. Train the robot links with N_FUNC = 8 for speed
    # print("Training robot links with N_FUNC = 8...")
    # rdf.train_links(
    #     link_names=ROBOT_LINKS,
    #     n_func=8,
    #     iters=TRAIN_ITERS,
    #     robot_name=ROBOT_NAME,
    #     debug=False,
    # )
    
    # 2. Train the fork with N_FUNC = 16 for better geometric fidelity
    print("Training fork with N_FUNC = 16...")
    rdf.train_links(
        link_names=FORK_LINK,
        n_func=24,
        iters=TRAIN_ITERS,
        robot_name=ROBOT_NAME,
        debug=False,
    )

if __name__ == '__main__':
    main()
