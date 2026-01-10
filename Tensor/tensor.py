from msilib import knownbits
from unittest import result
import numpy as np

#constants for memory calculations
BYTES_PER_FLOAT32 = 4 #memory to be allocated to varibles of type float32

#conversion metric to be used in conversion of KB to bytes
KB_TO_BYTES = 1024

#conversion metric to be used in conversion of MB to bytes
MB_TO_BYTES = 1024*1024


#class the performs the core ML operations
class Tensor():
    def __init__(self,data):
        """Creating a new tensor from data"""

        #1.Converting data to Numpy array with dtype=float32
        self.data = np.array(data,dtype=np.float32)
        #2. Setting self.shape from the array's shape
        self.shape = self.data.shape
        #3. Setting self.size from the array's size
        self.size = self.data.size
        #4. Setting self.dtype from the array's size
        self.dtype= self.data.dtype


    def __repr__(self):
        """String representation of a tensor for debugging"""
        return f"Tensor(data={self.data}),shape={self.shape}"

    def __str__(self):
        """Human readable string representation"""
        return f"Tensor({self.data})"

    def numpy(self):
        """Return the underlying Numpy array"""
        return self.data

    def memory_footprint(self):
        """Calculates exact memory usage in bytes"""
        return self.data.nbytes

    def __add__(self,other):
        """Add two tensors element-wise with broadcasting supporting"""

        #checking is other is a tensor
        if isinstance(other,Tensor):
            return Tensor(self.data + other.data)
        ##applying broadcasting
        else:
            return Tensor(self.data + other)


    def __sub__(self,other):
        """Subtract two tensors elementwise"""

        #checking if other is a tensor
        if isinstance(other,Tensor):
            return Tensor(self.data - other.data)
        else:
            return Tensor(self.data - other)

    def __mul__(self,other):
        """Multiplying two tensors elemetwise which not the same as matrix multiplication"""

        ##checking if other is a tensor
        if isinstance(other,Tensor):
            return Tensor(self.data*other.data)
        else:
            return Tensor(self.data*other)

    def __truediv__(self,other):
        """Divide two tensors element-wise"""

        if isinstance(other,Tensor):
            return Tensor(self.data / other.data)
        else:
            return Tensor(self.data / other)

    def matmul(self,other):
        """Performing matrix multiplication of two tensors"""
        if  not isinstance(other,Tensor):
            raise TypeError(f"Expected Tensor for matrix multiplication, got{type(other)}")
        #checking for scalar cases
        if self.shape ==() or other.shape ==():
            return Tensor(self.data*other.data)
        if len(self.shape) == 0 or len(other.shape) == 0:
            return Tensor(self.data*other.data)
        
         ##checking for 2D+ matrices
        if len(self.shape) >= 2 and len(other.shape) >= 2:
            if self.shape[-1] != other.shape[-2]:
                raise ValueError(
                    f"Cannot perform matrix multiplication:{self.shape} @ {other.shape}"
                    f"Inner dimensions must match: {self.shape[-1]} not equal to {other.shape[-2]}"

                )

        a = self.data
        b = other.data

        #handling 2D matrices with loops
        if len(a.shape) == 2 and len(b.shape) == 2:
            M, K = a.shape
            K2, N = b.shape
            result_data = np.zeros((M,N),dtype=a.dtype)

            for i in range(M):
                for j in range(N):
                    #dot product of row i from A with column j from B
                    result_data[i,j] =  np.dot(a[i,:],b[:,j]) 
        else:
            #for batches operation tensors with 3 or above dimensions
            result_data = np.matmul(a,b)

        return Tensor(result_data)

    def __matmul__(self,other):
        """Enabling @ operator for matmul"""
        return self.matmul(other)

    def __getitem__(self,key):
        """Enabling indexing and slicing operations of tensors"""

        result_data = self.data[key]

        if not isinstance(result_data,np.ndarray):
            result_data = np.array(result_data)

        return Tensor(result_data)


    def reshape(self,*shape):
        """Reshaping tensor to new dimensions"""

        if len(shape) == 1 and isinstance(shape[0],(tuple,list)):
            new_shape = tuple(shape[0])
        else:
            new_shape = shape
        if -1 in new_shape:
            if new_shape.count(1) > 1:
                raise ValueError("Can only specify one unknown dimension with -1")
            known_size=1
            unknown_idx = new_shape.index(-1)
            for i,dim in enumerate(new_shape):
                if i != unknown_idx:
                    known_size *= dim
            unknown_dim = self.size // known_size
            new_shape = list(new_shape)
            new_shape[unknown_idx] = unknown_dim
            new_shape = tuple(new_shape)

        if np.prod(new_shape) != self.size:
            target_size = int(np.prod(new_shape))
            raise ValueError(
                f"Total elements must match: {self.size} not equal to {target_size}"

            )
        reshaped_data =np.reshape(self.data,new_shape)
        return Tensor(reshaped_data) 

    def transpose(self,dim0=None,dim1=None):
        """Transpose tensor dimensions."""

        if dim0 is None and dim1 is None:
            ## returning a copy of the tensor
            ##since data has one dimension
            if len(self.shape) < 2:
                return Tensor(self.data.copy())

            ##swapping the specified dimensions
            else:
                axes = list(range(len(self.shape)))
                axes[-2], axes[-1] = axes[-1], axes[-2]
                transposed_data = np.transpose(self.data, axes)
        else:
            if dim0 is None or dim1 is None:
                raise ValueError("Both dim0 and dim1 must be specified")
            axes = list(range(len(self.shape)))
            axes[dim0],axes[dim1] = axes[dim1],axes[dim0]
            transposed_data = np.transpose(self.data,axes)
        return Tensor(transposed_data)


    def sum(self,axis=None,keepdims=False):
        """Summing a tensor along a specified axis"""

        result =np.sum(self.data,axis=axis,keepdims=keepdims)
        return Tensor(result)

    def mean(self,axis=None,keepdims=False):
        """Calculating mean along a specified axis"""
        result = np.mean(self.data,axis=axis,keepdims=keepdims)
        return Tensor(result)

    def max(self,axis=None,keepdims=False):
        """Finding maximum values along a specified axis"""
        result = np.max(self.data,axis=axis,keepdims=keepdims)
        return Tensor(result)
    


##testing 
def  tensor_creation():
    """Testing tensor creating with various data types."""

    #testing scalar scalar creation
    scalar = Tensor(5.0)
    assert scalar.data == 5.0
    assert scalar.shape == ()
    assert scalar.size == 1 
    assert scalar.dtype == np.float32

    #testing vector creation

    vector = Tensor([1,2,3])
    assert np.array_equal(vector.data,np.array([1,2,3],dtype=np.float32))
    assert vector.shape == (3,)
    assert vector.size == 3


    ##testing matrix creation
    matrix = Tensor([[1,2],[3,5]])
    assert np.array_equal(matrix.data,np.array([[1,2],[3,5]],dtype=np.float32))
    assert matrix.shape == (2,2)
    assert matrix.size == 4

    #test 3D tensor creation
    tensor_3d = Tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
    assert tensor_3d.shape == (2,2,2)
    assert tensor_3d.size == 8

    print("Tensor creation works correctly")

if __name__ == "__main__":
    tensor_creation()


