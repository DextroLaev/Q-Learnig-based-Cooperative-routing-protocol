k = 1.5
def absorption_coefficient(f):
    first_part = (0.11*(f**2))/(1+(f**2))
    second_part = (44*(f**2))/(4100+(f**2))
    third_part = (2.75*(f**2))/10000
    y_f = first_part+second_part+third_part+3/1000
    return y_f

def attenuation(d):
    f = (200/d)**(2/3)
    alpha = 10**(absorption_coefficient(f)/10)
    u_d = ((1000*d)**k)*(alpha**d)
    return u_d

def cooperative_energy_consumption(d_ij,d_cj,delta):
    first_part = attenuation(d_ij) + delta*(attenuation(d_cj) if d_cj > 0 else 0)
    return first_part/(1+delta)    

def lowest_transmission_power(po,d):
    return po*attenuation(d)