import numpy as np
import matplotlib.pyplot as plt

class Element:
    """
    The Element class keeps track of each element in the model, including cross-section properties, 
    element orientation (for coordinate system transformations), and the nodes that make up each element. 
    With the help of the Node class, it also keeps track of which Degrees of Freedom (DOFs) belong to each element.

    This class is responsible for providing the element stiffness matrix in the global coordinate system 
    (for subsequent assembly) and postprocessing element-level fields. 

    This class describes an element combining extension and Euler-Bernoulli bending. A similar (or inherited) 
    class could also be implemented for different element types (e.g., shear beam, Timoshenko beam, cable elements, etc). 
    For simplicity, it is assumed that elements are all arranged in a 2D plane.

    Attributes:
        node1 (Node): The first node of the element.
        node2 (Node): The second node of the element.
        EA (float): The axial stiffness of the element.
        EI (float): The flexural stiffness of the element.

    Methods:
        clear(): Clears the counting of elements.
        __init__(self, nodes): Initializes an Element object.
        set_section(self, props): Sets the section properties of the element.
        global_dofs(self): Returns the global degrees of freedom associated with the element.
        stiffness(self): Calculate the stiffness matrix of the element.
        add_distributed_load(self, q): Adds a distributed load to the element.
        bending_moments(self, u_global, num_points=2): Calculate the bending moments along the element.
        full_displacement(self, u_global, num_points=2): Calculates the displacement along the element.
        plot_moment_diagram(self, u_elem, num_points=10, global_c=False, scale=1.0): Plots the bending moment diagram of the element.
        plot_displaced(self, u_elem, num_points=10, global_c=False, scale=1.0): Plots the displaced element.
        __str__(self): Returns a string representation of the Element object.
    """

    ne = 0

    def clear():
        """
        Clears the counting of elements

        This method resets the class-level counters for number of elements. 
        It should be used when you want to start a new problem from scratch.
        """
        Element.ne = 0
        
    def __init__(self, node1, node2):
        """
        Initializes an Element object.

        Parameters:
        - node1 (Node): The first node of the element.
        - node2 (Node): The second node of the element.

        Attributes:
        - nodes (list): A list of Node objects representing the nodes of the element.
        - L (float): Length of the element.
        - cos (float): Cosine of the element's orientation angle.
        - sin (float): Sine of the element's orientation angle.
        - T (ndarray): Transformation matrix.
        - Tt (ndarray): Transpose of the transformation matrix.

        Returns:
        None
        """
        self.nodes = [node1, node2]

        self.L = np.sqrt((self.nodes[1].x - self.nodes[0].x)**2.0 + (self.nodes[1].z - self.nodes[0].z)**2.0)

        # Make use of numpy.arctan2 to return the angle between -pi() and pi()
        alpha = np.arctan2 (self.nodes[1].z - self.nodes[0].z, self.nodes[1].x - self.nodes[0].x)

        T = np.zeros((6, 6))

        T[0, 0] = T[1, 1] = T[3, 3] = T[4, 4] = np.cos(alpha)
        T[0, 1] = T[3, 4] = np.sin(alpha)
        T[1, 0] = T[4, 3] = -np.sin(alpha)
        T[2, 2] = T[5, 5] = 1.0

        self.T = T
        self.Tt = np.transpose(T)

        self.q = np.array([0,0])
        self.local_element_load = np.array([0,0,0,0,0,0])
        
        Element.ne += 1

    def set_section(self, props):
        """
        Sets the section properties of the element.

        Parameters:
        - props (dict): A dictionary containing the section properties.
                        The dictionary should have the following keys:
                        - 'EA': The axial stiffness of the element.
                        - 'EI': The flexural stiffness of the element.

        Returns:
        None
        """
        if 'EA' in props:
            self.EA = props['EA']
        else:
            self.EA = 1.e20
        if 'EI' in props:
            self.EI = props['EI']
        else:
            self.EI = 1.e20

    def global_dofs(self):
        """
        Returns the global degrees of freedom associated with the element.

        Returns:
            numpy.ndarray: Array containing the global degrees of freedom.
        """
        return np.hstack((self.nodes[0].dofs, self.nodes[1].dofs))

    def stiffness(self):
        """
        Calculate the stiffness matrix of the element.

        Returns:
        np.ndarray: The stiffness matrix of the element.
        """
        k = np.zeros((6, 6))

        EA = self.EA
        EI = self.EI
        L = self.L

        # Axial stiffness terms
        k[0, 0] = k[3, 3] = EA / L
        k[0, 3] = k[3, 0] = -EA / L
        
        # Bending stiffness terms
        k[1, 1] = k[4, 4] = 12 * EI / (L**3)
        k[1, 2] = k[2, 1] = 6 * EI / (L**2)
        k[1, 4] = k[4, 1] = -12 * EI / (L**3)
        k[1, 5] = k[5, 1] = 6 * EI / (L**2)
        
        k[2, 2] = k[5, 5] = 4 * EI / L
        k[2, 4] = k[4, 2] = -6 * EI / (L**2)
        k[2, 5] = k[5, 2] = 2 * EI / L
        
        k[4, 5] = k[5, 4] = -6 * EI / (L**2)

        return np.matmul(np.matmul(self.Tt, k), self.T)

    def add_distributed_load(self, q):
        """
        Adds a distributed load to the element.

        Parameters:
            q (list): List of distributed load in local x and z direction [q_axial, q_transverse].

        Returns:
            None
        """

        l = self.L
        self.q = np.array(q)

        # Compute equivalent nodal loads for constant distributed load
        # For axial load q[0]: equally distributed to both nodes
        # For transverse load q[1]: fixed-fixed beam equivalent loads
        self.local_element_load = np.array([
            q[0] * l / 2,              # axial force at node 1
            q[1] * l / 2,               # shear force at node 1
            q[1] * l**2 / 12,           # moment at node 1
            q[0] * l / 2,               # axial force at node 2
            q[1] * l / 2,               # shear force at node 2
            -q[1] * l**2 / 12           # moment at node 2 (negative for consistency)
        ])

        # Transform local element loads to global coordinate system 
        global_element_load = np.matmul(self.Tt, self.local_element_load)

        # Add the equivalent loads to the nodes
        # Node 1 gets the first 3 DOFs (axial, shear, moment)
        self.nodes[0].add_load(global_element_load[0:3])
        # Node 2 gets the last 3 DOFs (axial, shear, moment)
        self.nodes[1].add_load(global_element_load[3:6])

    def bending_moments(self, u_global, num_points=2):
        """
        Calculate the bending moments along the element.

        Parameters:
        - u_global (numpy.ndarray): Global displacement vector.
        - num_points (int): Number of points to evaluate the bending moments. Default is 2.

        Returns:
        - M (numpy.ndarray): Array of bending moments at the specified points.
        """

        l = self.L
        q = self.q[1]
        EI = self.EI

        # Transform global displacements to local coordinate system
        local_disp = np.matmul(self.T, u_global)
        
        # Extract local displacements
        u1 = local_disp[0]  # axial displacement at node 1
        w1 = local_disp[1]  # transverse displacement at node 1
        theta1 = local_disp[2]  # rotation at node 1
        u2 = local_disp[3]  # axial displacement at node 2
        w2 = local_disp[4]  # transverse displacement at node 2
        theta2 = local_disp[5]  # rotation at node 2

        # Points along the element
        local_x = np.linspace(0.0, l, num_points)
        M = np.zeros(num_points)

        # For a beam with constant distributed load q, the bending moment is:
        # M(x) = -EI * d²w/dx²
        # Using the cubic Hermite shape functions for w(x):
        # w(x) = w1*N1 + theta1*N2 + w2*N3 + theta2*N4 + particular solution for distributed load
        # The particular solution for constant q is: (q/(24EI)) * x² * (x - l)²
        
        for i, x in enumerate(local_x):
            # Bending moment from Hermite shape functions (second derivatives)
            # Second derivatives of shape functions:
            # N1''(x) = (12x - 6l)/l³
            # N2''(x) = (6x - 4l)/l²
            # N3''(x) = (-12x + 6l)/l³
            # N4''(x) = (6x - 2l)/l²
            
            M_hermite = EI * (
                w1 * (12*x - 6*l) / (l**3) +
                theta1 * (6*x - 4*l) / (l**2) +
                w2 * (-12*x + 6*l) / (l**3) +
                theta2 * (6*x - 2*l) / (l**2)
            )
            
            # Particular solution for constant distributed load
            # M_particular = -EI * (d²/dx² of particular solution)
            # For w_particular = (q/(24EI)) * x² * (x - l)²
            # M_particular = -(q/2) * (x² - l*x + l²/6)
            M_particular = -(q/2) * (x**2 - l*x + l**2/6)
            
            M[i] = M_hermite + M_particular

        return M
    
    def full_displacement(self, u_global, num_points=2):
        """
        Calculates the displacement along the element.

        Args:
            u_global (numpy.ndarray): Global displacement vector of the element.
            num_points (int, optional): Number of points to calculate the bending moments. Default is 2.

        Returns:
            numpy.ndarray: Array of displacement along the element.
        """
        
        l = self.L
        q_axial = self.q[0]
        q_trans = self.q[1]
        EA = self.EA
        EI = self.EI

        # Transform global displacements to local coordinate system
        local_disp = np.matmul(self.T, u_global)
        
        # Extract local displacements
        u1 = local_disp[0]  # axial displacement at node 1
        w1 = local_disp[1]  # transverse displacement at node 1
        theta1 = local_disp[2]  # rotation at node 1
        u2 = local_disp[3]  # axial displacement at node 2
        w2 = local_disp[4]  # transverse displacement at node 2
        theta2 = local_disp[5]  # rotation at node 2

        # Points along the element
        local_x = np.linspace(0.0, l, num_points)
        u = np.zeros(num_points)  # axial displacement
        w = np.zeros(num_points)  # transverse displacement

        for i, x in enumerate(local_x):
            # Axial displacement (linear interpolation + particular solution for distributed axial load)
            # For constant axial load q_axial, particular solution: (q_axial/(2EA)) * x * (l - x)
            u[i] = u1 * (1 - x/l) + u2 * (x/l) + (q_axial/(2*EA)) * x * (l - x)
            
            # Transverse displacement using Hermite shape functions + particular solution for distributed transverse load
            # Hermite shape functions:
            N1 = 1 - 3*(x/l)**2 + 2*(x/l)**3
            N2 = x * (1 - x/l)**2
            N3 = 3*(x/l)**2 - 2*(x/l)**3
            N4 = x * ((x/l)**2 - x/l)
            
            # Particular solution for constant transverse load q_trans
            # w_particular = (q_trans/(24*EI)) * x² * (x - l)²
            w_particular = (q_trans/(24*EI)) * x**2 * (x - l)**2
            
            w[i] = w1 * N1 + theta1 * N2 + w2 * N3 + theta2 * N4 + w_particular

        return u, w
    
    def plot_moment_diagram (self, u_elem, num_points=10, global_c=False, scale=1.0):
        """
        Plots the bending moment diagram of the element.

        Args:
            u_global (numpy.ndarray): Global displacement vector of the element.
            num_points (int, optional): Number of points to calculate the bending moments. Default is 2.
            global_c (bool, optional): If True, plots the bending moment diagram in the global coordinate system. Default is False (plots in local coordinate system).
            scale (float, optional): Scale factor for the bending moment diagram. Default is 1.0.

        Returns:
            None
        """
        import matplotlib.pyplot as plt

        x = np.linspace ( 0.0, self.L, num_points )
        M = self.bending_moments ( u_elem, num_points )
        xM_local = np.vstack((np.hstack([0,x,x[-1]]),np.hstack([0,M,0])*scale))
        if global_c:
            xM_global = np.matmul(self.Tt[0:2,:2],xM_local)
            xz_start_node = np.vstack((np.ones(num_points+2)*self.nodes[0].x, np.ones(num_points+2)*self.nodes[0].z))
            xz_Mlijn = xM_global + xz_start_node
            p = plt.plot(xz_Mlijn[0,:],xz_Mlijn[1,:])
            X0= self.nodes[0].x
            Z0= self.nodes[0].z
            X1= self.nodes[1].x
            Z1= self.nodes[1].z
            plt.plot((X0, X1), (Z0, Z1), color=p[0].get_color())
            plt.axis('off')
            plt.axis('equal')
        else:
            p = plt.plot(xM_local[0,:],xM_local[1,:])
            plt.xlabel ( "x" )
            plt.ylabel ( "M" )
        if not plt.gca().yaxis_inverted():
            plt.gca().invert_yaxis()
        plt.gcf().patch.set_alpha(0.0)
        plt.gca().patch.set_alpha(0.0)
        plt.gca().patch.set_alpha(0.0)
        plt.title('Moment line')

    def plot_displaced(self, u_elem, num_points=10, global_c=False, scale=1.0):
        """
        Plots the displacd element.

        Args:
            u_global (numpy.ndarray): Global displacement vector of the element.
            num_points (int, optional): Number of points to calculate the bending moments. Default is 2.
            global_c (bool, optional): If True, plots the displacement diagram in the global coordinate system. Default is False (plots in local coordinate system).
            scale (float, optional): Scale factor for the displacement diagram. Default is 1.0.

        Returns:
            None
        """

        x = np.linspace ( 0.0, self.L, num_points )
        u, w = self.full_displacement ( u_elem, num_points )
        uw_local = np.vstack((x+u*scale,w*scale))
        if global_c:
            uw_global = np.matmul(self.Tt[:2,:2],uw_local)
            xz_start_node = np.vstack((np.ones(num_points)*self.nodes[0].x, np.ones(num_points)*self.nodes[0].z))
            uw = uw_global + xz_start_node
            p =  plt.plot(uw[0,:],uw[1,:])
            X0= self.nodes[0].x
            Z0= self.nodes[0].z
            X1= self.nodes[1].x
            Z1= self.nodes[1].z
            plt.plot((X0, X1), (Z0, Z1), color=p[0].get_color(),alpha=0.3)
            plt.axis('off')
            plt.axis('equal')
        else:
            p = plt.plot(uw_local[0,:],uw_local[1,:])
            plt.plot((0, self.L), (0, 0), color=p[0].get_color(),alpha=0.3)
        if not plt.gca().yaxis_inverted():
            plt.gca().invert_yaxis()
        plt.gcf().patch.set_alpha(0.0)
        plt.gca().patch.set_alpha(0.0)
        plt.gca().patch.set_alpha(0.0)
        plt.title('Displaced structure')

    def plot_numbered_structure(self,beam_number):
        """
        Plots the nodes and elements of the structure with their node and element numbers.

        Returns:
            None
        """

        X0= self.nodes[0].x
        Z0= self.nodes[0].z
        X1= self.nodes[1].x
        Z1= self.nodes[1].z
        node_num = []
        node_num.append(self.nodes[0].dofs[0] // 3)
        node_num.append(self.nodes[1].dofs[0] // 3)
        plt.plot((X0, X1), (Z0, Z1), color='black',alpha=0.3)
        for i, node in enumerate(self.nodes):
            plt.text(node.x, node.z, f'[{node.dofs[0] // 3}]', fontsize=12, ha='center', va='center')
        plt.text((X0+X1)/2, (Z0+Z1)/2, f'({beam_number})', fontsize=12, ha='center', va='center')
        if not plt.gca().yaxis_inverted():
            plt.gca().invert_yaxis()
        plt.axis('off')
        plt.axis('equal')
        plt.gcf().patch.set_alpha(0.0)
        plt.gca().patch.set_alpha(0.0)
        plt.gca().patch.set_alpha(0.0)


    def __str__(self):
        """
        Returns a string representation of the Element object.
        
        The string includes the values of the node1, node2 attributes.
        """
        return f"Element connecting:\nnode #1:\n {self.nodes[0]}\nwith node #2:\n {self.nodes[1]}"