def mysqrt(x): return np.sqrt((1.+0j)*x)

def a2case2():
    aux0=Q*((np.cos(Δϕ))*((np.cos(θ1))*((np.cos(θ2))*((np.sin(θ1))*(np.sin(θ2))))));
    aux1=((chip**2)*(((np.cos(θ1))**2)))-((chieff**2)*((((1.+q)**2))*((((np.sin(Δϕ))**2))*(((np.sin(θ1))**2)))));
    aux2=((chip**2)*((q**2)*((((np.cos(θ2))**2))*(((np.sin(θ1))**2)))))+((-2.*((chip**2)*(q*aux0)))+((Q**2)*(aux1*(((np.sin(θ2))**2)))));
    aux3=(chieff*(q*((1.+q)*((np.cos(θ2))*(((np.sin(θ1))**2))))))-(mysqrt(((((np.cos(θ1))**2))*aux2)));
    aux4=chieff*(q*(Q*((np.cos(Δϕ))*((np.cos(θ1))*((np.sin(θ1))*(np.sin(θ2)))))));
    aux5=(aux3-aux4)-(chieff*(Q*((np.cos(Δϕ))*((np.cos(θ1))*((np.sin(θ1))*(np.sin(θ2)))))));
    aux6=Q*((np.cos(Δϕ))*((np.cos(θ1))*((np.cos(θ2))*((np.sin(θ1))*(np.sin(θ2))))));
    aux7=((q**2)*((((np.cos(θ2))**2))*(((np.sin(θ1))**2))))+((-2.*(q*aux6))+((Q**2)*((((np.cos(θ1))**2))*(((np.sin(θ2))**2)))));
    output=aux5/aux7
    return output
