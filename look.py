import qiskit
from qiskit import QuantumCircuit , QuantumRegister
from qiskit.circuit import Parameter
import matplotlib.pyplot as plt
print(qiskit.version.get_version_info())
qc = QuantumCircuit(1)
x1,x2, theta1, theta2=Parameter("x1"), Parameter("x2"), Parameter("θ1"), Parameter("θ2")

qc.h(0)
qc.ry(x1,0)
qc.rz(x2,0)

qc.rx(theta1, 0)
qc.rz(theta2, 0)
qc.measure_all()
qc.draw('mpl')  # matplotlib output, good for thesis
plt.show()
