# Author: - Achmadya Ridwan Ilyawan
#         - Fahmi Ahmad Fadilah
#         - Hilman Permana
# Date: 2023-04-02
# Description: Konversi warna citra dari RGB ke CIELUV

import cv2
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st


# konversi warna RGB ke CIELUV
def RGB2LUV(I):
    Ixyz = cv2.cvtColor(I, cv2.COLOR_RGB2XYZ)
    Ix = Ixyz[:,:,0]
    Iy = Ixyz[:,:,1]
    Iz = Ixyz[:,:,2]
    [m,n] = Ix.shape
    
    # white point d65
    xn = 0.95047
    yn = 1.00000
    zn = 1.08883
    
    # konversi XYZ ke L*u*v*
    for i in range(m):
        for j in range(n):
            x = Ix[i,j]/xn
            y = Iy[i,j]/yn
            z = Iz[i,j]/zn
            
            if x > 0.008856:
                x = x**(1/3)
            else:
                x = (7.787*x) + (16/116)
            
            if y > 0.008856:
                y = y**(1/3)
            else:
                y = (7.787*y) + (16/116)
            
            if z > 0.008856:
                z = z**(1/3)
            else:
                z = (7.787*z) + (16/116)
            
            L = (116*y) - 16
            u = 13*L*(x-y)
            v = 13*L*(y-z)
            
            Ix[i,j] = L
            Iy[i,j] = u
            Iz[i,j] = v
    
    I = cv2.merge((Ix,Iy,Iz))
    return I

# judul halaman web
st.title('Konversi Warna Citra dari RGB ke CIELUV')

# upload gambar
uploaded_file = st.file_uploader("Upload Files", type=["jpg", "png", "jpeg"])

# menampilkan gambar
if uploaded_file is not None:

    image = np.frombuffer(uploaded_file.read(), np.uint8)
    image = cv2.imdecode(image, cv2.IMREAD_COLOR)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # menampilkan gambar
    colimg1, colimg2 = st.columns(2)

    with colimg1:
        st.image(image, caption='Gambar Asli', use_column_width=True)

    # konversi warna
    with colimg2:
        image = RGB2LUV(image)
        st.image(image, caption='Gambar Hasil', use_column_width=True)



# menampilkan kode sumber
tab1, tab2 = st.tabs(['MATLAB', 'Python'])

with tab1:
    st.markdown('''
    ```matlab
    function f = RGB2LUV(I)
        Ixyz = rgb2xyz(I);
        Ix = Ixyz(:,:,1);
        Iy = Ixyz(:,:,2);
        Iz = Ixyz(:,:,3);
        [m,n] = size(Ix);
        
        % white point d65
        xn = 0.95047;
        yn = 1;
        zn = 1.08883;
        
        for i = 1 : m
            for j = 1 : n
                Li = double(Iy(i,j)/yn);
                if Li > 0.008856
                    IL(i,j) = (116*nthroot(Li,3)-16);
                else
                    IL(i,j) = (903.3*Li);
                end
                u = 4*Ix(i,j)/(Ix(i,j)+15*Iy(i,j)+3*Iz(i,j));
                un = 4*xn/(xn+15*yn+3*zn);
                v = 9*Iy(i,j)/(Ix(i,j)+15*Iy(i,j)+3*Iz(i,j));
                vn = 9*yn/(xn+15*yn+3*zn);
                Iu(i,j) = (13*IL(i,j)*(u-un));
                Iv(i,j) = (13*IL(i,j)*(v-vn));
            end
        end
        ILuv(:,:,1) = IL;
        ILuv(:,:,2) = Iu;
        ILuv(:,:,3) = Iv;
        
        % figure(1), imshow(ILuv);
        f = ILuv;
    end
    ```
    ''', unsafe_allow_html=True)


with tab2:
    st.markdown('''
    ```python
    def RGB2LUV(I):
        Ixyz = cv2.cvtColor(I, cv2.COLOR_RGB2XYZ)
        Ix = Ixyz[:,:,0]
        Iy = Ixyz[:,:,1]
        Iz = Ixyz[:,:,2]
        [m,n] = Ix.shape
        
        # white point d65
        xn = 0.95047
        yn = 1.00000
        zn = 1.08883
        
        # konversi XYZ ke L*u*v*
        for i in range(m):
            for j in range(n):
                x = Ix[i,j]/xn
                y = Iy[i,j]/yn
                z = Iz[i,j]/zn
                
                if x > 0.008856:
                    x = x**(1/3)
                else:
                    x = (7.787*x) + (16/116)
                
                if y > 0.008856:
                    y = y**(1/3)
                else:
                    y = (7.787*y) + (16/116)
                
                if z > 0.008856:
                    z = z**(1/3)
                else:
                    z = (7.787*z) + (16/116)
                
                L = (116*y) - 16
                u = 13*L*(x-y)
                v = 13*L*(y-z)
                
                Ix[i,j] = L
                Iy[i,j] = u
                Iz[i,j] = v
        
        I = cv2.merge((Ix,Iy,Iz))
        return I
    ```
    ''', unsafe_allow_html=True)