# Author: - Achmadya Ridwan Ilyawan
#         - Fahmi Ahmad Fadilah
#         - Hilman Permana
# Date: 2023-04-02
# Description: Konversi warna citra dari RGB ke NTSC

import cv2
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

# konversi warna RGB ke NTSC
def RGB2NTSC(I):
        Ir = I[:,:,0]
        Ig = I[:,:,1]
        Ib = I[:,:,2]
        m, n = Ir.shape

        k = np.array([[0.299, 0.587, 0.114],
                    [0.516, -0.274, -0.322],
                    [0.211, -0.523, 0.312]])

        for i in range(m):
            for j in range(n):
                r = Ir[i,j]
                g = Ig[i,j]
                b = Ib[i,j]

                y = k[0,0]*r + k[0,1]*g + k[0,2]*b
                u = k[1,0]*r + k[1,1]*g + k[1,2]*b
                v = k[2,0]*r + k[2,1]*g + k[2,2]*b

                Ir[i,j] = y
                Ig[i,j] = u
                Ib[i,j] = v

        I = np.dstack((Ir, Ig, Ib))
        return I


# judul halaman web
st.title('Konversi Warna Citra dari RGB ke NTSC')

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
        image = RGB2NTSC(image)
        st.image(image, caption='Gambar Hasil Konversi', use_column_width=True)



# menampilkan kode sumber
tab1, tab2 = st.tabs(['MATLAB', 'Python'])

with tab1:
    st.markdown('''
    ```matlab
    function f = RGB2NTSC(I)
        Ir = I(:,:,1);
        Ig = I(:,:,2);
        Ib = I(:,:,3);
        [m,n] = size(Ir);

        k = [0.299 0.587 0.114;
            0.516 -0.274 -0.322;
            0.211 -0.523 0.312;];
        
        for i = 1 : m
        for j = 1 : n
            rgb = [Ir(i,j); Ig(i,j); Ib(i,j)];
            yiq = k*double(rgb);
            Iy(i,j) = double(yiq(1,:)/255);
            Ii(i,j) = double(yiq(2,:)/255);
            Iq(i,j) = double(yiq(3,:)/255);
        end
        end
        IYiq(:,:,1) = Iy;
        IYiq(:,:,2) = Ii;
        IYiq(:,:,3) = Iq;
        
        % figure(1), imshow(IYiq);
        f = IYiq;
    end
    ```
    ''', unsafe_allow_html=True)


with tab2:
    st.markdown('''
    ```python
    def RGB2NTSC(I):
        Ir = I[:,:,0]
        Ig = I[:,:,1]
        Ib = I[:,:,2]
        m, n = Ir.shape

        k = np.array([[0.299, 0.587, 0.114],
                    [0.516, -0.274, -0.322],
                    [0.211, -0.523, 0.312]])

        for i in range(m):
            for j in range(n):
                r = Ir[i,j]
                g = Ig[i,j]
                b = Ib[i,j]

                y = k[0,0]*r + k[0,1]*g + k[0,2]*b
                u = k[1,0]*r + k[1,1]*g + k[1,2]*b
                v = k[2,0]*r + k[2,1]*g + k[2,2]*b

                Ir[i,j] = y
                Ig[i,j] = u
                Ib[i,j] = v

        I = np.dstack((Ir, Ig, Ib))
        return I

    ```
    ''', unsafe_allow_html=True)