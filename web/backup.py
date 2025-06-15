import streamlit as st
import pandas as pd
from dotenv import load_dotenv

# Import thư viện LIDA
from lida import Manager, TextGenerationConfig, llm

# import hàm hỗ trợ từ thư mục helpers
from helpers.helpers import (
    upload_file, 
    clean_df, 
    load_api_key, 
    base64_to_image, 
    display_charts
)

# Load environment variables
load_dotenv()

def initialize_lida_and_cohere(api_key):
    """
    Khởi tạo LIDA và Cohere với cấu hình mặc định
    
    Agrs:
        api_key (object): mã API của người dùng trên nền tảng Cohere
        
    """
    try:
        # Khởi tạo Cohere với API Key và thiết lập với LIDA
        text_gen = llm("cohere", api_key=api_key)
        lida = Manager(text_gen=text_gen)
        
        # Cấu hình text generation cho Cohere
        with st.expander("Filter"):
            temperature = st.slider("Temperature", min_value=0.0, max_value=1.0, value=0.5, step=0.1)
            model = st.selectbox("Model", ["command-xlarge-nightly",
                                           "command-large", 
                                           "command-base-nightly"])
            textgen_config = TextGenerationConfig(
                n=1, 
                temperature=temperature, 
                model=model,
                use_cache=True
            )
        
        st.sidebar.success("Kết nối với Cohere thành công!")
        return lida, textgen_config
    
    except Exception as e:
        st.sidebar.error(f"Không thể kết nối với Cohere: {e}")
        return None, None



def process_data_summary(df, lida, textgen_config):
    """
    Thực hiện tóm tắt và đặt mục tiêu cho dữ liệu
    
    Args:
        df (dataframe): bộ dữ liệu đầu vào
        lida : thư viện lida đóng vai trò là người xử lý các tasks
        textgen_config: cấu hình của TextGeneration đối với LIDA
    
    Returns:
        Hình ảnh các biểu đồ dựa trên summary mà người dùng yêu cầu
    """
    st.dataframe(df.head())
    
    # Kiểm tra và làm sạch dữ liệu
    null_values = df.isnull().sum()
    dup_values = df.duplicated().sum()
    
    if null_values.any() > 0 or dup_values > 0:  
        df = clean_df(df)
        st.success("Đã làm sạch dữ liệu!")
    else:
        st.success("Dữ liệu không có giá trị rỗng/trùng lặp.")
    if len(df.columns) >= 10:
        st.warning(
                    "Lưu ý: LIDA hoạt động tốt nhất với bộ dữ liệu có dưới 10 cột. "
                    "Nếu bộ dữ liệu của bạn có hơn 10 cột, LIDA vẫn sẽ chạy nhưng tạo ra kết quả không đẹp mắt, hữu ích."
                    )
    
    summary = lida.summarize(df, summary_method="default", textgen_config=textgen_config)
    goals = lida.goals(summary, n=5, textgen_config=textgen_config)
    
    st.subheader("Bảng tóm lược:")
    st.write(summary)
    
    st.subheader("Goals:")
    for goal in goals:
        st.write(goal)
    
    return summary, goals

def generate_visualizations(lida, summary, goals, textgen_config):
    """
    Tạo và hiển thị các biểu đồ theo mục tiêu
    Args:
        df (dataframe): bộ dữ liệu đầu vào
        lida : thư viện lida đóng vai trò là người xử lý các tasks
        textgen_config: cấu hình của TextGeneration đối với LIDA
    
    Returns:
        Hình ảnh các biểu đồ dựa trên summary mà người dùng yêu cầu
    """
    
    library = "seaborn"
    n = 5
    
    for i in range(n):
        try:
            charts = lida.visualize(summary=summary, goal=goals[i], library=library)
            
            for chart in charts:
                st.subheader(f"Goal {i+1}")
                display_charts(
                    lida, 
                    chart, 
                    goals[i], 
                    library, 
                    textgen_config
                )
        
        except Exception as e:
            st.error(f"Lỗi khi tạo biểu đồ cho Goal {i+1}: {e}")

def process_user_query_graphs(df, lida, textgen_config):
    """
    Tạo biểu đồ dựa trên truy vấn người dùng
    
    Args:
        df (dataframe): bộ dữ liệu đầu vào
        lida : thư viện lida đóng vai trò là người xử lý các tasks
        textgen_config: cấu hình của TextGeneration đối với LIDA
    
    Returns:
        Hình ảnh các biểu đồ dựa trên summary mà người dùng yêu cầu
        
    """
    user_query = st.text_area(label="User Query:")
    k = st.number_input(label="Số biểu đồ bạn muốn tạo:", min_value=1, max_value=5, step=1)
    
    if st.button("Tạo biểu đồ"):
        try:
            textgen_config.n = k
            summary = lida.summarize(df, summary_method="default", textgen_config=textgen_config)
            query_charts = lida.visualize(summary=summary, goal=user_query, textgen_config=textgen_config)
            
            st.write(f"Số biểu đồ gợi ý: {len(query_charts)}")
            for chart in query_charts:
                display_charts(
                    lida, 
                    chart, 
                    user_query, 
                    "seaborn", 
                    textgen_config
                )
            if len(query_charts) < k:
                    st.error(f"Xin lỗi. Chúng tôi hiện tại chỉ có thể gợi ý cho bạn {len(query_charts)} biểu đồ.")
                    
        except Exception as e:
            st.error(f"Đã xảy ra lỗi khi khởi tạo LIDA hoặc Cohere: {e}")

def process_viz_recommend(df, lida, textgen_config):
    """
    Đề xuất biểu đồ dựa trên tóm tắt dữ liệu
    
    Args:
        df (dataframe): bộ dữ liệu đầu vào
        lida : thư viện lida đóng vai trò là người xử lý các tasks
        textgen_config: cấu hình của TextGeneration đối với LIDA
    
    Returns:
        Hình ảnh các biểu đồ dựa trên summary mà người dùng yêu cầu
    """
    
    k = st.number_input(label="Số biểu đồ bạn muốn tạo:", min_value=1, max_value=5, step=1)
    
    if st.button(label="Tạo biểu đồ"):
        try:
            textgen_config_recommend = TextGenerationConfig(
                n=1, 
                temperature=0.3,
                use_cache=True
            )
            
            summary = lida.summarize(df, summary_method="default", textgen_config=textgen_config)
            goals = lida.goals(summary, n=1, textgen_config=textgen_config)
            
            charts = lida.visualize(summary=summary, goal=goals[0], library="seaborn")
            
            if charts:
                textgen_config_recommend.n = k
                recommended_charts = lida.recommend(
                    code=charts[0].code, 
                    summary=summary, 
                    n=k,  
                    textgen_config=textgen_config
                )
                
                st.write(f"Số biểu đồ gợi ý: {len(recommended_charts)}")
                for chart in recommended_charts:
                    display_charts(
                        lida, 
                        chart, 
                        goals[0], 
                        "seaborn", 
                        textgen_config
                    )
            else:
                st.error("Không có biểu đồ nào được tạo.")
        
        except Exception as e:
            st.error(f"Lỗi trong quá trình đề xuất biểu đồ: {e}")

def show_task():
    """
    Chính của ứng dụng, điều phối các task
    """
    # Load API key
    api_key = load_api_key()
    
    # Khởi tạo LIDA và Cohere
    lida, textgen_config = initialize_lida_and_cohere(api_key) if api_key else (None, None)

    with st.sidebar.container():
        st.header("Tasks:")
        task = st.selectbox("Các chức năng:",["Summarize & Goal", 
                                            "UserQuery based graphs",
                                            "VizRecommend"
                                            ])

    # Nội dung tương ứng với từng lựa chọn
    df = upload_file()
    
    if df is not None:  
        if task == "Summarize & Goal" and lida:
            summary, goals = process_data_summary(df, lida, textgen_config)
            generate_visualizations(lida, summary, goals, textgen_config)
        
        elif task == "UserQuery based graphs" and lida:
            process_user_query_graphs(df, lida, textgen_config)
        
        elif task == "VizRecommend" and lida:
            process_viz_recommend(df, lida, textgen_config)



show_task()







import streamlit as st
from streamlit_option_menu import option_menu

def show_home():
    # Tạo menu với option_menu
    # selected = option_menu( 
    #     menu_title="NTViz Menu",
    #     options=["Home", "LIDA", "Contribute", "Source", "Support"],  # Các mục
    #     icons=["house", "bar-chart", "hand-thumbs-up", "book", "envelope"],  # Các biểu tượng
    #     menu_icon="cast",  # Biểu tượng menu
    #     default_index=0,  # Mục mặc định
    #     orientation="horizontal"  # Menu ngang
    # )

    home, lida, contr, source, contact = st.tabs(["Home", "LIDA", "Contribute", "Source", "Support"])
    # Hiển thị nội dung dựa trên mục đã chọn
    with home:
        st.title(" :violet[NTViz] _:gray[A Data Recommendation Systems For EveryOne]❤️!_")
        st.subheader(":grey[Tại sao chúng tôi xây dựng dự án này?]")
        st.markdown(""" 
                    _:blue[**"Dữ liệu là loại tiền tệ có giá trị nhất"**]_, là điều mà ta không thể nào phủ nhận được vì
                    chúng có các vai trò chủ chốt trong nhiều ngành nghề, lĩnh vực như:
                    - **Hỗ trợ ra quyết định:** cung cấp thông tin và phân tích để giúp tổ chức và cá nhân đưa ra các quyết định đáng tin cậy, từ chiến lược kinh doanh đến chi tiết về sản phẩm và dịch vụ.
                    - **Nâng cao hiệu quả hoạt động:** giúp tối ưu hóa hoạt động bằng cách cung cấp thông tin về hiệu suất, quy trình và khách hàng, giúp tổ chức điều chỉnh và cải thiện hoạt động để đạt hiệu quả cao hơn.
                    - **Phát triển sản phẩm và dịch vụ:** cung cấp thông tin về nhu cầu và phản hồi từ thị trường, giúp các doanh nghiệp hiểu rõ hơn về mong đợi của khách hàng và phát triển các sản phẩm và dịch vụ phù hợp.
                    - **Nghiên cứu và phát triển công nghệ:** là nguồn tài nguyên quan trọng cho các nghiên cứu khoa học và phát triển công nghệ, giúp nhà nghiên cứu phân tích và tạo ra các khám phá mới.
                    - **Tăng cường trải nghiệm khách hàng:** giúp cá nhân hóa trải nghiệm khách hàng, từ sản phẩm đến dịch vụ chăm sóc khách hàng, dựa trên thông tin cá nhân.                    
                    - **Giảm thiểu rủi ro và gian lận:** giúp phát hiện và ngăn chặn các hành vi gian lận và rủi ro trong kinh doanh và tài chính, thông qua phân tích các mô hình và xu hướng không bình thường. \n
                    ...và còn nhiều lợi ích khác từ dữ liệu có thể mang lại cho chúng ta. \n
                    Với số lượng dữ liệu ngày càng đa dạng và phức tạp, ta không thể nào hiểu chúng hết chỉ bằng cách đọc các dữ liệu thô được thu thập từ thực tế.
                    Với số lượng dữ liệu ngày càng đa dạng và phức tạp, việc chỉ đọc các dữ liệu thô thu thập từ thực tế không thể giúp chúng ta hiểu hết được giá trị của chúng. Làm thế nào để nắm bắt thông tin một cách nhanh chóng và dễ dàng nhất?\n
                    :point_right: Đáp án chính là :violet[**Trực quan hóa dữ liệu**], một công cụ mạnh mẽ giúp chúng ta chuyển hóa dữ liệu thành những hình ảnh dễ hiểu, từ đó đưa ra quyết định chính xác và hiệu quả hơn bao giờ hết. 
                    """)
        st.image("material/outlook/mhoa.jpg", caption="Hình 1. Minh họa",  use_container_width=True)    



        # Lí do vì sao trực quan hóa dữ liệu quan trọng
        st.markdown(""" 
                ### :grey[Mức độ quan trọng của :violet[Trực Quan Hóa Dữ Liệu]]:
                **1. Truyền tải thông tin một cách hiệu quả:**
                - Trực quan hóa giúp biểu diễn dữ liệu phức tạp dưới dạng hình ảnh dễ hiểu, giúp người xem nắm bắt nhanh thông tin mà không cần phân tích sâu các bảng số liệu.

                **2. Hỗ trợ ra quyết định:**
                - Các biểu đồ và hình ảnh trực quan giúp người ra quyết định nhận diện xu hướng, phát hiện vấn đề và đưa ra giải pháp phù hợp dựa trên dữ liệu cụ thể.

                **3. Phát hiện xu hướng và mối quan hệ:**
                - Trực quan hóa giúp làm nổi bật các xu hướng, mẫu hình (patterns), và mối quan hệ ẩn giữa các biến trong dữ liệu mà có thể bị bỏ qua khi chỉ nhìn vào dữ liệu thô.

                **4. Giao tiếp và thuyết phục:**
                - Các biểu đồ và hình ảnh trực quan hỗ trợ việc thuyết phục người khác, đặc biệt là trong các bài thuyết trình, báo cáo, hoặc tranh luận dựa trên dữ liệu.

                **5. Nâng cao khả năng phân tích dữ liệu:**
                - Giúp các nhà phân tích khám phá dữ liệu sâu hơn, phát hiện các giá trị bất thường (outliers) hoặc các khía cạnh chưa từng được xem xét, từ đó đưa ra những phân tích toàn diện hơn.
                
                **6. Tăng khả năng tương tác với dữ liệu:**
                -  Tương tác với dữ liệu thông qua hình ảnh, và các thông tin số liệu quan trọng như doanh số, phân phối,...
                
                **7. Tạo cảm giác hứng thú với dữ liệu:**
                - Hình ảnh sinh động và trực quan không chỉ làm cho dữ liệu bớt khô khan mà còn giúp người không chuyên có thể tìm hiểu sâu hơn về dữ liệu và các vấn đề liên quan 1 cách đơn giản, dễ hiểu hơn.
                
                """)
        st.image("material/outlook/ex_tquan.jpg", caption="Hình 2. Ví dụ Trực Quan")
        
        
        
        st.divider()
        st.markdown("### :grey[Bài Toán Chính:]")
        st.markdown("""
                    Tuy nhiên, nếu 1 người không có các kỹ năng lập trình muốn tìm hiểu sâu, rút trích thông tin từ bộ dữ liệu của mình thì họ sẽ vấp phải các khó khăn như:
                    - Nên trực quan hóa theo biến nào? Biểu đồ đó có ý nghĩa gì?
                    - Nên chọn biểu đồ nào để trực quan hóa cho bộ dữ liệu?
                    - Làm thế nào để hiện thực hóa biểu đồ đó khi không có kỹ năng lập trình? \n
                    Nhận thấy được điều đó, chúng tôi ấp ủ kế hoạch và tìm hiểu các tool có thể hỗ trợ :blue["Những người không chuyên về dữ liệu"] một cách đơn giản, và ít tốn kém nhất.
                    ##### :point_right: Bằng cách xây dựng :violet[Hệ Thống Gợi Ý Biểu Đồ] dựa trên bộ dữ liệu của người dùng.
                    """)
    

    with lida:
        st.header(":grey[LIDA: A Tool for Automatic Generation of Grammar-Agnostic Visualizations and Infographics using Large Language Models]")
        st.markdown("### *Tổng quan về LIDA:*")
        st.markdown(""" 
                    - LIDA là nền tảng tạo biểu đồ dựa trên LLM, thuộc sở hữu của Microsoft, gồm 4 module chính: Summarize, Goal Explorer, VisGenerator và Infographic
                    - Hệ thống sử dụng LLM để tóm tắt dữ liệu, tạo mục tiêu, sinh mã và tạo biểu đồ tự động
                    """)
        st.markdown(""" 
                    ##### Ưu điểm:
                    - Tự động tạo giả thuyết/mục tiêu từ dữ liệu, hỗ trợ nhiều ngữ pháp trực quan, và có khả năng tạo infographic
                    - Hiệu quả hơn các hệ thống hiện có, đơn giản hóa quá trình tạo biểu đồ phức tạp
                    - Giới thiệu các chỉ số đánh giá độ tin cậy (VER) và chất lượng trực quan hóa (SEVQ)
                    """)
        st.markdown(""" 
                    ##### Nhược điểm:
                    - Cần nghiên cứu thêm về tác động của độ phức tạp tác vụ và lựa chọn ngôn ngữ lập trình đến hiệu suất
                    - Yêu cầu nhiều tài nguyên tính toán, cần cải thiện về triển khai và độ trễ
                    - Cần phát triển các tiêu chuẩn đánh giá toàn diện hơn và nghiên cứu về khả năng giải thích hành vi của hệ thống
                    """)
        
        st.markdown("### *Chi tiết các hoạt động của LIDA:*")
        st.markdown(""" 
        1. **Summarize and Goals:** \n
            *a. Summarize dựa theo rule và LLM:*
            -    Dùng LLM để tạo nên một mô tả ngắn, cô đọng về cái tập dataset qua 2 stage process mà người dùng bỏ vào nhằm định hướng được mục tiêu, hoặc gợi ý, ví dụ như đối với tập dữ liệu này thì mình có thể làm được những gì, như thế nào. \n
            *b. Goal Explorer:*
            -    Nhìn chung trong step này, nó sẽ tạo ra 1 tập file .JSON gồm 3 đối tượng: “question”, “visualization”, “rationale”
            - Question:
                - LLM đóng vai trò như người dùng, người hướng dẫn, tự bản thân nó sẽ đi khám phá, tìm hiểu về tập dữ liệu này để đưa ra các giả thuyết như là các câu hỏi
            - Visualization:
                - Tên và loại biểu đồ
            - Rationale:
                - Biểu đồ mang ý nghĩa như thế nào? Đưa ra những "insight" gì?        
        """)
        st.image("material/lida/goals.png", caption="Hình 2. Cấu trúc của Goals",
                use_container_width=True)
        
        st.markdown("""
        2. **VisGenerator:**
        Tạo ra biểu đồ cụ thể, thực hiện dựa theo 3 module con: \n 
            *a. Code scaffold constructor:* \n
            Tiến hành thư viện mã scafffolds tương ứng với ngôn ngữ lập trình Scaffolds support `Matplotlib`, `GGPlot`, `Plotly`, `Altair`, `Seaborn`, và `Boken`
            
            *b. Code generator:* \n
            Lấy scaffold, bộ dữ liệu mà ta đã tóm tắt, mục tiêu trực quan và prompt mà ta đã dựng sẵn đưa vào LLM
            
            *c. Code executor:*  
            
            Thực hiện vẽ, tạo biểu đồ cụ thể.
        """)
        st.image("material/lida/example.png", caption="Hình 3. Biểu đồ tương ứng", 
                use_container_width=True)   
        st.markdown("""
        3. **Infographic:**
            - Module này được sử dụng để tạo ra những cái đồ thị dựa trên kết quả đầu ra của VisGenerator
            - Sử dụng text-conditioned image-to-image ( khả năng tạo ảnh từ văn bản trong các mô hình khuếch tán) (Rombach và cộng sự, 2022), được triển khai qua API của thư viện Peacasso (Dibia, 2022).
            """)
        
        st.image("material/lida/infograp.png", 
                caption="Hình 4: Minh họa infographic của LIDA",
                use_container_width=True)
        
        st.divider()
        
        # Thông tin LIDA hỗ trợ các nền tảng nào
        
        st.markdown("### _:grey[📌 Những lưu ý quan trọng:]_")
        st.markdown(""" 
                    ##### 1. Python:
                    LIDA yêu cầu Python từ version 3.10 trở lên.
                    ##### 2. Dữ liệu: 
                    Phù hợp nhất với tập dữ liệu có <= 10 cột. Đối với dữ liệu lớn hơn, cần xử lý trước (chọn cột phù hợp).
                    ##### 3. Khả năng hoạt động: 
                    LIDA yêu cầu dữ liệu ở định dạng như .csv hoặc .json (danh sách đối tượng).
                    ##### 4. Hiệu quả: 
                    LIDA hoạt động tốt hơn với các LLM lớn (GPT-3.5, GPT-4). Các mô hình nhỏ hơn có thể không theo sát hướng dẫn tốt.
                    ##### 5. Độ chính xác: 
                    Tỷ lệ lỗi < 3.5% trên 2200 biểu đồ được tạo, thấp hơn mức cơ bản (>10%).
                    ##### 6. Large Language Model: 
                    LIDA sử dụng các mô hình ngôn ngữ lớn như 1 người quản lý giúp người dùng sử dụng các tác vụ của nó.
                    - OpenAI (sử dụng bằng cách kéo API KEY từ trang web)
                    - COHERE (là nền tảng chính mà chúng tôi sử dụng)
                    - Các LLM trên HUGGING FACE như là: [microsoft/Phi-3-mini-128k-instruct](https://huggingface.co/microsoft/Phi-3-mini-128k-instruct) hoặc [meta-llama/Llama-3.2-3B-Instruct](https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct)
                        - Lưu ý: 
                            - Nếu như máy tính của bạn không có GPU thì chúng tôi khuyên bạn hãy chạy trên các phần mềm thay thế như Google Colab, Kaggle.
                            - Nếu bạn muốn trải nghiệm các model trên HUGGING FACE thì các model cần đạt các điều kiện sau:
                                1. Được huấn luyện với tác vụ **text-generation**, đưa tập dữ liệu về định dạng **.json** thì mới hợp lệ.
                                2. **max_tokens_length > 1024**
                                3. **1B <=  Số parameters  <= 4B** (vì GPU P100 hoặc T4 chỉ có thể chạy được model có số parameters nhỏ hơn 4B)
                                4. Đặt thêm giới hạn cho số lượng tokens tạo mới, max_tokens cho nó, nếu không thì máy sẽ mất nhiều thời gian để có thể chạy ra kết quả.
                    """)

    with contr:
        st.subheader("_:grey[Sự Đóng góp của NTViz]_")
        st.markdown(""" 
                    Đề tài này đóng góp vào việc hỗ trợ những người làm trong các lĩnh vực cần nhiều thông tin, các tệp dữ liệu nhưng không chuyên, hoặc các bạn học sinh, sinh viên không có kiến thức nền về Phân Tích Dữ Liệu dễ dàng tiếp cận và trực quan hoá các tập tin của mình nhằm phục vụ cho mục đích của bản thân.
                    - **Hỗ trợ những người không chuyên về Phân tích Dữ liệu:** giúp làm sạch và dùng biểu đồ để trực quan hoá dữ liệu của họ.
                    - **Đề xuất các biểu đồ tự động:** giúp tối ưu hóa hiệu quả bằng cách tự động đề xuất các biểu đồ trực quan hóa theo các chủ đề mà người dùng muốn hướng tới.
                    - **Tiết kiệm thời gian:** Tự động hóa quy trình lựa chọn biểu đồ phù hợp, giúp người dùng tiết kiệm thời gian và giảm bớt việc thử-sai.
                    - **Cải thiện khả năng tiếp cận dữ liệu:** Giảm rào cản trong việc truy cập và trực quan hóa dữ liệu cho những người không có kỹ năng kỹ thuật, từ đó khuyến khích việc sử dụng dữ liệu trong quyết định kinh doanh hoặc nghiên cứu.
                    - **Công cụ học tập:** Từ việc đề xuất biểu đồ, hệ thống đóng vai trò như một công cụ học tập cho các học sinh, sinh viên hoặc người đi làm, giúp họ hiểu rõ các nguyên tắc trực quan, diễn giải, ý nghĩa của biểu đồ được đề xuất.   
                    - **Khả năng mở rộng và tích hợp:** Hệ thống có tiềm năng được mở rộng và tích hợp vào các nền tảng phân tích khác, giúp tối ưu hóa quy trình làm việc của nhiều doanh nghiệp và tổ chức.
                    """)
        
        st.divider()
        # Những cải tiến mới chúng tôi đề xuất
        st.subheader("_:grey[Cải tiến mới trong LIDA:]_")
        st.markdown(""" 
                    Sau khi tìm hiểu kĩ càng về các tác vụ trong LIDA, chúng tôi nhận thấy:
                    - Trong LIDA, khi nhập đầu vào bộ dữ liệu thì LIDA không có tính năng "Cleaning DATA" sẵn cho người dùng.
                    - Tính năng "summarize" của nó chưa đủ rõ ràng và trực diện. Ví dụ:
                    """)
        st.image("material/lida/summary.png", 
                caption="Hình 1. Ví dụ về bảng tóm tắt của 1 cột dữ liệu.",
                use_container_width=True)
        st.markdown(""" 
                    - Tính năng tạo biểu đồ dựa trên câu truy vấn có nhiều tiềm năng. Nó có thể tạo tốt từ 3 đến 5 biểu đồ chỉ dựa theo câu truy vấn mà người dùng đưa ra.
                    """)
        st.markdown("""
                    Do đó, chúng tôi đề xuất thêm:
                    - Tích hợp thư viện `ydata_profiling` để đưa ra những thông tin chung trực diện hơn cho bộ dữ liệu.
                    - Trước khi LIDA **"summarize"** và **"goals explorer"**, chúng tôi sẽ tích hợp thêm 1 hàm giúp người dùng kiểm tra dữ liệu đã sạch hay chưa, nếu chưa thì chúng tôi sẽ làm sạch giúp họ.
                    """)
        st.image("material/lida/overview.png", 
                caption="Hình 2. Minh Họa Overview",
                use_container_width=True)

    # Nguồn tham khảo
    with source:
        st.subheader("_:grey[Nguồn Tham Khảo và Ghi Nhận:]_")
        st.markdown("[LIDA: A Tool for Automatic Generation of Grammar-Agnostic Visualizations and Infographics using Large Language Models](https://aclanthology.org/2023.acl-demo.11) (Dibia, ACL 2023)")
        st.markdown("[COHERE, API KEY](https://dashboard.cohere.com/?_gl=1*5bkinq*_gcl_au*MTQ5NTExMTE3MS4xNzMwOTcxODg1*_ga*NTc0ODQ4NDk0LjE3MzA5NzE4ODI.*_ga_CRGS116RZS*MTczNDI0Njk5Ni4yMi4wLjE3MzQyNDY5OTYuNjAuMC4w)")
        st.write("_:violet[Chúng tôi xin trân trọng gửi lời cảm ơn chân thành đến :blue[LIDA] và :blue[COHERE] vì đã cung cấp các công cụ và dịch vụ tuyệt vời, giúp dự án này trở nên khả thi. Không có sự hỗ trợ của họ, chúng tôi không thể đạt được kết quả như hiện tại. ❤️]_")    

    with contact:
        st.subheader("📞 :grey[Liên hệ]")
        st.markdown("""
        - **Email**: support@ntviz.com
        - **Website**: [NTViz](https://ntviz.com)
        - **Hotline**: 0123-456-789
        """)


show_home()


import streamlit as st
import os
from dotenv import load_dotenv
import pandas as pd
# from dataprep.clean import clean_headers, clean_missing, clean_date, clean_columns

# Import thư viện LIDA
from lida import Manager, TextGenerationConfig, llm
import seaborn as sns
import matplotlib.pyplot as plt
import base64
from io import BytesIO
from PIL import Image

# Load environment variables
load_dotenv()

def load_api_key():
    
    """Load API Key from user input or environment variables."""
    if "api_key" not in st.session_state:
        st.session_state.api_key = ""
    
    # Input API key
    st.sidebar.subheader("Nhập mã API key:")
    api_key = st.sidebar.text_input(
        "API key:", value=st.session_state.api_key, type="password", placeholder="Nhập mã API của bạn"
    )
    st.session_state.api_key = api_key

    # Validate API key
    if not api_key:
        st.warning("Vui lòng nhập API key để tiếp tục.")
    else:
        pass
        # st.success("API key đã được nhập.")

    return api_key


def upload_file():
    """
    Hàm này dùng để:
    - Tải tệp dữ liệu với 2 định dạng csv và json
    - Đọc dữ liệu đầu vào cho các tác vụ tiếp theo
    """
    # Tải tệp lên để thực hiện các tác vụ
    uploaded_file = st.file_uploader("Tải tệp dữ liệu với định dạng .csv/.json:", type=["csv", "json"])
    if uploaded_file is not None:
        # Xử lý tệp đã tải lên
        if uploaded_file.name.endswith(".csv"):
            # Đọc file CSV
            df = pd.read_csv(uploaded_file)
            st.success(f"Đã tải tệp CSV có {len(df)} dòng dữ liệu.")
        elif uploaded_file.name.endswith(".json"):
            # Đọc file văn bản
            df = pd.read_json(uploaded_file)
            st.success(f"Đã tải tệp JSON.")
        else:
            st.error("Định dạng này không được hỗ trợ.")
        
        return df
    else:
        st.error("Không tồn tại tệp dữ liệu nào. Vui lòng thử lại.")
        return None


def clean_df(df):
    """
    Mục tiêu: Tự động hóa quá trình làm sạch data cho dữ liệu như thay thế giá trị rỗng, xóa dữ liệu trùng lặp
    
    Args:
        df (_type_): bộ dữ liệu người dùng đưa vào

    Returns:
        cleaned_df: bộ dữ liệu đã được xử lý
    """
    df = df.copy()
    
    # Điền giá trị thiếu cho các cột số bằng giá trị trung bình
    for col in df.select_dtypes(include=['number']).columns:
        df[col] = df[col].fillna(df[col].mean())
        
    # Xóa bỏ các giá trị trùng lặp
    df = df.drop_duplicates()
    
    return df

# Convert base64 string to image
def base64_to_image(base64_str):
    img_data = base64.b64decode(base64_str)
    img = Image.open(BytesIO(img_data))
    return img

# Hàm output, hiển thị code, explain, evalute biểu đồ
def display_charts(
    lida, 
    chart, 
    goal, 
    library='seaborn', 
    textgen_config=None
):
    """
    Hiển thị, giải thích và đánh giá chi tiết cho một biểu đồ.

    Args:
        lida (Manager): LIDA manager
        chart (object): Biểu đồ cần phân tích
        goal (str): Mục tiêu của biểu đồ
        library (str, optional): Thư viện vẽ biểu đồ. Mặc định là 'matplotlib'.
        textgen_config (TextGenerationConfig, optional): Cấu hình sinh văn bản. 
                                                         Mặc định là None.
    """
    # Kiểm tra nếu chart không tồn tại
    if not chart:
        st.warning("Không có biểu đồ để hiển thị.")
        return

    # Chuyển đổi biểu đồ sang ảnh
    try:
        img = base64_to_image(chart.raster)
    except Exception as e:
        st.error(f"Lỗi chuyển đổi biểu đồ: {e}")
        return

    # Hiển thị ảnh biểu đồ
    st.image(img)

    # Nút tải xuống
    buffered = BytesIO()
    img.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    href = f'<a href="data:image/png;base64,{img_str}" download="chart.png"> ˚⋆ ʕっ• ᴥ • ʔっ Download Chart ˚⋆ </a>'
    st.markdown(href, unsafe_allow_html=True)

    # Hiển thị code
    with st.expander("Code"):
        st.code(chart.code, language="python")

    # Giải thích biểu đồ
    with st.expander("Explain"):
        try:
            explanations = lida.explain(
                code=chart.code, 
                library=library, 
                textgen_config=textgen_config
            )
            for row in explanations[0]:
                st.write(row["section"], " ** ", row["explanation"])
        except Exception as e:
            st.error(f"Không thể giải thích biểu đồ: {e}")

    # Đánh giá biểu đồ
    with st.expander("Evaluate"):
        try:
            evaluations = lida.evaluate(
                code=chart.code,
                goal=goal,
                textgen_config=textgen_config,
                library=library
            )[0]
            
            for eval in evaluations:
                st.write(
                    eval["dimension"],
                    "Score",
                    eval["score"],
                    "/ 10"
                )
                st.write(eval["rationale"])
        except Exception as e:
            st.error(f"Không thể đánh giá biểu đồ: {e}")
            
            
        
# Set up textgen config
def setup_model_textgen(models, provider):
    filter, requirements = st.tabs(["👩‍🏫 Filter Instruction", "🛑 Requirements"])
    # Configure text generation for Cohere
    with filter:
        with st.popover("Instruction:"):
            st.markdown(""" 
                        Set the temperature (creativity level):
                        - Lower values (e.g., 0.1–0.3): The model produces simple and predictable results.
                        - Higher values (e.g., 0.7–1.0): The model becomes more creative and complex, but the results might be less focused.
                        - Recommended range: Choose a temperature between 0.5 and 0.7 for a good balance between accuracy and creativity.
                                """)
            if provider == "Cohere":
                st.markdown(""" 
                                Choose the model you would like to use for your dataset:
                                - "command-xlarge-nightly": The largest model with the most details, able to learn complex patterns and relationships in data.
                                - "command-large: Smaller than xlarge but still powerful, offering a good mix of performance and efficiency.
                                - "command-base-nightly" : The smallest and lightest model, designed for quick use and easy deployment.
                            """)
            if provider == "Gemini":
                st.markdown(""" 
                                Choose the model you would like to use for your dataset:
                                - "Gemini 2.0 Flash-exp:" The largest and most powerful model in the Gemini family, designed for complex tasks that require a high level of creativity and understanding.
                                - "Gemini 1.5 Flash:" This model offers a good balance between size and performance. It can handle a wide range of tasks and is suitable for many applications.
                                - "Gemini 1.5 Flash-8B:" The smallest model in the Gemini family, designed for simpler tasks and devices with limited resources.
                                - "Gemini 1.5 Pro:" This model is optimized for specific tasks, such as code generation or technical translation.
                            """)
    with requirements:
            st.markdown(""" 
                    **NTViz works best with the datasets:**
                    - Columns: ≤ 10  
                    - Rows: ≤ 1000  
                    - File Size: ≤ 500KB  

                    **Correct CSV Format:**  
                    Your file should contain only the variable names as column headers and the corresponding values.  
                    Avoid including unrelated information such as titles, notes, or daily reports in the file.  
                    """)
                    
            with st.popover("📋 Example:"):
                    st.markdown(""" 
                        **✅ Correct Input Example:**
                        ```csv
                        Name, Age, Country, Salary
                        John, 25, USA, 50000
                        Anna, 30, Canada, 60000
                        Mark, 28, UK, 45000
                        ```

                        **❌ Incorrect Input Example:**
                        ```csv
                        ,Daily Report,,
                        Date,12/16/2024 20:38,,
                        Tên,Phiến,,
                        ,,,
                        ID,Name,Category,Date
                        1,A,BCV,12/16/2024 20:38
                        2,B,B1,12/16/2024 20:38
                        3,A,B2,12/16/2024 20:38
                        ```
                        - Extra rows like "Daily Report" or empty cells will cause errors.  
                        - Remove unnecessary information before uploading.  
                        """)
                    
    # Choose the model and the temperature
    # Model and temperature selection
    temperature = st.slider("Temperature", 
                            min_value=0.0, 
                            max_value=1.0, 
                            value=0.5, 
                            step=0.1)
    model = st.selectbox("Select Model:", models)
    textgen_config = TextGenerationConfig(
                    n=1, 
                    temperature=temperature, 
                    model=model,
                    use_cache=True
                )
    return textgen_config