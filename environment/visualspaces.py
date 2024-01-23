import numpy as np
from gym import spaces

class visualspaces(spaces.Space):
    
    def __init__(self):
        """
        Initialize the amorphous space.

        Parameters
        ----------
        - spheres : list
            A list of dictionaries representing the spherical regions in the 
            space. Each dictionary should contain the following keys:
          - 'center' : list 
                The center of the sphere (a 3D numpy array).
          - 'radius' : float
                The radius of the sphere.
        """

        #spheres = [{'center': np.array([-0.1, -0.07, 0.0]), 'radius': 0.03},
        #           {'center': np.array([-0.08, -0.07, 0.0]), 'radius': 0.023},
        #          # Add more spheres as needed
        #           ]
        spheres = []

        # Generate 80 spheres with random centers and radii
        for _ in range(80):
            center = np.random.uniform(low=-0.5, high=0.5, size=(3,))
            radius = np.random.uniform(low=0.01, high=0.1)
            spheres.append({'center': center, 'radius': radius})



        self.spheres = spheres
        self.low = np.array([sphere['center'] - sphere['radius'] for sphere in spheres]).flatten()
        self.high = np.array([sphere['center'] + sphere['radius'] for sphere in spheres]).flatten()
        super(visualspaces, self).__init__((3,))

    def sample(self):
        """Sample a random point from the amorphous space."""
        # Choose a random sphere
        sphere = self.spheres[np.random.randint(len(self.spheres))]

        # Generate a random point within the sphere
        phi = np.random.uniform(low=0, high=2*np.pi)
        theta = np.random.uniform(low=0, high=np.pi)
        r = np.random.uniform(low=0, high=sphere['radius'])
        x = sphere['center'][0] + r * np.sin(theta) * np.cos(phi)
        y = sphere['center'][1] + r * np.sin(theta) * np.sin(phi)
        z = sphere['center'][2] + r * np.cos(theta)
        
        return np.array([x, y, z])

    def contains(self, x):
        """Check if a point is within the bounds of the amorphous space."""
        for sphere in self.spheres:
            if np.linalg.norm(x - sphere['center']) <= sphere['radius']:
                return True
        return False

    def clip(self, x):
        """Clip a point to the bounds of the amorphous space."""
        if self.contains(x):
            return x
        else:
            # Find the nearest point on the boundary of the space
            min_distance = float('inf')
            nearest_point = None
            for sphere in self.spheres:
                distance = np.linalg.norm(x - sphere['center'])
                if distance < min_distance:
                    min_distance = distance
                    nearest_point = sphere['radius'] * (x - sphere['center']) / distance + sphere['center']
            return nearest_point
        