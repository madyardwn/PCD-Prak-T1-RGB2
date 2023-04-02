# Author: - Achmadya Ridwan Ilyawan
#         - Fahmi Ahmad Fadilah
#         - Hilman Permana
# Date: 2023-04-02
# Description: Konversi warna citra dari RGB ke YUV

import cv2
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

# konversi warna RGB ke YUV
import numpy as np

import numpy as np

import numpy as np

def RGB2YUV(I):
        Ir = I[:,:,0]
        Ig = I[:,:,1]
        Ib = I[:,:,2]
        [m,n] = Ir.shape
        
        # konversi RGB ke YUV
        for i in range(m):
            for j in range(n):
                r = Ir[i,j]
                g = Ig[i,j]
                b = Ib[i,j]
                
                y = (0.299*r) + (0.587*g) + (0.114*b)
                u = (-0.147*r) - (0.289*g) + (0.436*b)
                v = (0.615*r) - (0.515*g) - (0.100*b)
                
                Ir[i,j] = y
                Ig[i,j] = u
                Ib[i,j] = v
        
        I = cv2.merge((Ir,Ig,Ib))
        return I


# judul halaman web
st.title('Konversi Warna Citra dari RGB ke YUV')

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
        image = RGB2YUV(image)
        st.image(image, caption='Gambar Hasil', use_column_width=True)



# menampilkan kode sumber
tab1, tab2 = st.tabs(['MATLAB', 'Python'])

with tab1:
    st.markdown('''
    ```matlab
    function f = RGB2YUV(I)
        Ir = I(:,:,1);
        Ig = I(:,:,2);
        Ib = I(:,:,3);
        [m,n] = size(Ir);
        
        k = [0.299 0.587 0.114;
            -0.147 -0.289 0.436;
            0.615 -0.515 -0.100;];
        
        for i = 1 : m
        for j = 1 : n
            rgb = [Ir(i,j); Ig(i,j); Ib(i,j)];
            yuv = k*double(rgb);
            Iy(i,j) = yuv(1,:);
            Iu(i,j) = yuv(2,:);
            Iv(i,j) = yuv(3,:);
        end
        end
        IYuv(:,:,1) = Iy;
        IYuv(:,:,2) = Iu;
        IYuv(:,:,3) = Iv;
        
        % figure(1), imshow(IYuv);
        f = IYuv;
    end
    ```
    ''', unsafe_allow_html=True)


with tab2:
    st.markdown('''
    ```python
    def RGB2YUV(I):
        Ir = I[:,:,0]
        Ig = I[:,:,1]
        Ib = I[:,:,2]
        [m,n] = Ir.shape
        
        # konversi RGB ke YUV
        for i in range(m):
            for j in range(n):
                r = Ir[i,j]
                g = Ig[i,j]
                b = Ib[i,j]
                
                y = (0.299*r) + (0.587*g) + (0.114*b)
                u = (-0.147*r) - (0.289*g) + (0.436*b)
                v = (0.615*r) - (0.515*g) - (0.100*b)
                
                Ir[i,j] = y
                Ig[i,j] = u
                Ib[i,j] = v
        
        I = cv2.merge((Ir,Ig,Ib))
        return I
    ```
    ''', unsafe_allow_html=True)