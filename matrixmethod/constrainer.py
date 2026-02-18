import numpy as np

class Constrainer:
    """
    A class that represents a constrainer for fixing degrees of freedom in a structural analysis.

    Attributes:
        cons_dofs (list): A list of constrained degrees of freedom.
        cons_vals (list): A list of corresponding constraint values.

    Methods:
        fix_dof: Fixes a degree of freedom at a specific value.
        fix_node: Fixes all degrees of freedom of a node.
        full_disp: Combines the displacements of free and constrained degrees of freedom.
        constrain: Applies the constraints to the stiffness matrix and load vector.
        support_reactions: Calculates the support reactions based on the constrained displacements.
    """

    def __init__(self):
        """
        Initializes a new instance of the Constrainer class.

        Attributes:
            cons_dofs (list): A list of constrained degrees of freedom.
            cons_vals (list): A list of corresponding constraint values.  
        """
        self.cons_dofs = []
        self.cons_vals = []

    def fix_dof (self, node, dof, value = 0):
        """
        Fixes a degree of freedom at a specific value.

        Args:
            node (Node): The node object.
            dof (int): The index of the degree of freedom to fix.
            value (float, optional): The value to fix the degree of freedom at. Defaults to 0.
        """
        self.cons_dofs.append(node.dofs[dof])
        self.cons_vals.append(value)
 
    def fix_node (self, node):
        """
        Fixes all degrees of freedom of a node.

        Args:
            node (Node): The node object.
        """
        for dof in [0,1,2]:
            self.fix_dof (node, dof)    

    def full_disp (self,u_free):
        """
        Combines the displacements of free and constrained degrees of freedom.

        Args:
            u_free (numpy.ndarray): The displacements of the free degrees of freedom.

        Returns:
            numpy.ndarray: The combined displacements of all degrees of freedom.
        """
        u_full = np.zeros(len(self.free_dofs) + len(self.cons_dofs))
        
        u_full[self.free_dofs] = u_free
        u_full[self.cons_dofs] = self.cons_vals
        
        return u_full
    
    def constrain (self, k, f):
        """
        Applies the constraints to the stiffness matrix and load vector.

        Args:
            k (numpy.ndarray): The stiffness matrix.
            f (numpy.ndarray): The load vector.

        Returns:
            tuple: A tuple containing the stiffness matrix corresponding to free dofs and the corresponding load vector.
        """
        self.free_dofs = [i for i in range(len(f)) if i not in self.cons_dofs]
        
        Kff = k[np.ix_(self.free_dofs,self.free_dofs)]
        Kfc = k[np.ix_(self.free_dofs,self.cons_dofs)]
        Ff = f[self.free_dofs]

        return Kff, Ff - np.matmul(Kfc,self.cons_vals)

    def support_reactions (self,k,u_free,f):       
        """
        Calculates the support reactions based on the constrained displacements.

        Args:
            k (numpy.ndarray): The stiffness matrix.
            u_free (numpy.ndarray): The displacements of the free degrees of freedom.
            f (numpy.ndarray): The load vector.

        Returns:
            numpy.ndarray: The support reactions.
        """
        Kcf = k[np.ix_(self.cons_dofs,self.free_dofs)]
        Kcc = k[np.ix_(self.cons_dofs,self.cons_dofs)]
        
        return np.matmul(Kcf,u_free) + np.matmul(Kcc,self.cons_vals) - f[self.cons_dofs]

    def __str__(self):
        """
        Returns a string representation of the Constrainer object.

        Returns:
            str: A string representation of the Constrainer object.
        """
        return f"This constrainer has constrained the degrees of freedom: {self.cons_dofs} with corresponding constrained values: {self.cons_vals})"
