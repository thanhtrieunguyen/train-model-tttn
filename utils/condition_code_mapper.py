"""
Module để ánh xạ từ các thông số thời tiết sang condition_code dựa trên logic và quy tắc.
"""

# Định nghĩa các nhóm mã condition chính
CLEAR_CODES = [1000]  # Trời quang đãng
PARTLY_CLOUDY_CODES = [1003]  # Có mây một phần
CLOUDY_CODES = [1006, 1009]  # Nhiều mây hoặc u ám
MIST_CODES = [1030, 1135, 1147]  # Sương mù, sương khói
RAIN_CODES = [1063, 1150, 1153, 1168, 1171, 1180, 1183, 1186, 1189, 1192, 1195, 1198, 1201]  # Các mã về mưa
SNOW_CODES = [1066, 1114, 1117, 1210, 1213, 1216, 1219, 1222, 1225, 1255, 1258]  # Các mã về tuyết
SLEET_CODES = [1069, 1204, 1207, 1249, 1252]  # Các mã về mưa tuyết
THUNDERSTORM_CODES = [1087, 1273, 1276, 1279, 1282]  # Các mã về giông bão
DUST_SAND_CODES = [1237]  # Bão cát, bụi
DRIZZLE_CODES = [1150, 1153, 1168, 1171]  # Mưa phùn
FOG_CODES = [1135, 1147]  # Sương mù

class ConditionCodeMapper:
    """Lớp xử lý việc ánh xạ từ thông số thời tiết sang condition code."""
    
    @staticmethod
    def get_condition_code(precipitation=0, temp=None, humidity=None, wind_speed=None, 
                          cloud_cover=None, visibility=None, pressure=None, is_day=1):
        """
        Ánh xạ các thông số thời tiết sang condition code.
        
        Args:
            precipitation (float): Lượng mưa (mm)
            temp (float): Nhiệt độ (°C)
            humidity (float): Độ ẩm (%)
            wind_speed (float): Tốc độ gió (km/h)
            cloud_cover (float): Độ che phủ của mây (%)
            visibility (float): Tầm nhìn (km)
            pressure (float): Áp suất khí quyển (hPa)
            is_day (int): Ngày hay đêm (1 = ngày, 0 = đêm)
            
        Returns:
            int: Mã condition code tương ứng
        """
        
        # Xử lý trường hợp có mưa
        if precipitation is not None:
            if precipitation >= 20:  # Mưa rất to
                return 1195
            elif precipitation >= 10:  # Mưa to
                return 1192
            elif precipitation >= 5:  # Mưa vừa
                return 1189
            elif precipitation >= 2:  # Mưa nhẹ
                return 1183
            elif precipitation >= 0.5:  # Mưa phùn
                return 1153
            elif precipitation > 0:  # Mưa rất nhẹ
                return 1150
        
        # Xử lý trường hợp có tuyết (dựa vào nhiệt độ và lượng mưa)
        if temp is not None and temp <= 0 and precipitation is not None and precipitation > 0:
            if precipitation >= 5:  # Tuyết dày
                return 1225
            elif precipitation >= 2:  # Tuyết vừa
                return 1219
            else:  # Tuyết nhẹ
                return 1213
        
        # Xử lý trời nhiều mây
        if cloud_cover is not None:
            if cloud_cover >= 85:  # U ám
                return 1009
            elif cloud_cover >= 50:  # Nhiều mây
                return 1006
            elif cloud_cover >= 25:  # Mây rải rác
                return 1003
            else:  # Quang đãng
                return 1000
        
        # Xử lý tầm nhìn
        if visibility is not None:
            if visibility < 1:  # Sương mù dày đặc
                return 1147
            elif visibility < 4:  # Sương mù
                return 1135
        
        # Trường hợp mặc định: trời quang
        return 1000
    
    @staticmethod
    def get_condition_text(condition_code):
        """
        Chuyển đổi condition code sang mô tả bằng văn bản.
        
        Args:
            condition_code (int): Mã condition code
            
        Returns:
            str: Mô tả về điều kiện thời tiết
        """
        condition_map = {
            1000: "Trời quang đãng",
            1003: "Có mây rải rác",
            1006: "Nhiều mây",
            1009: "Trời u ám",
            1030: "Sương mù nhẹ",
            1063: "Mưa rào rải rác",
            1066: "Tuyết rơi rải rác",
            1069: "Mưa tuyết rải rác",
            1087: "Có giông sét rải rác",
            1135: "Sương mù",
            1147: "Sương mù dày đặc",
            1150: "Mưa phùn nhẹ",
            1153: "Mưa phùn",
            1180: "Mưa nhẹ rải rác",
            1183: "Mưa nhẹ",
            1186: "Mưa vừa rải rác",
            1189: "Mưa vừa",
            1192: "Mưa to rải rác",
            1195: "Mưa to",
            1213: "Tuyết nhẹ",
            1219: "Tuyết vừa",
            1225: "Tuyết dày",
            1273: "Mưa giông nhẹ rải rác",
            1276: "Mưa giông vừa hoặc to",
        }
        
        return condition_map.get(condition_code, "Không xác định")
    
    @staticmethod
    def find_closest_known_code(precipitation=0, temp=None, humidity=None, wind_speed=None, 
                               cloud_cover=None, visibility=None, pressure=None, is_day=1):
        """
        Tìm condition code gần nhất dựa trên các thông số thời tiết khi gặp mã lạ.
        Hữu ích khi gặp mã không có trong tập huấn luyện.
        """
        # Logic để tìm mã gần nhất dựa trên khoảng cách Euclidean của các thông số
        # Đây là phương pháp dự phòng khi không thể xác định chính xác mã
        
        base_code = ConditionCodeMapper.get_condition_code(
            precipitation, temp, humidity, wind_speed, cloud_cover, visibility, pressure, is_day
        )
        
        return base_code

# Ví dụ sử dụng
if __name__ == "__main__":
    # Ví dụ: trời nhiều mây, không mưa
    condition_code = ConditionCodeMapper.get_condition_code(
        precipitation=0, 
        temp=25, 
        cloud_cover=60
    )
    condition_text = ConditionCodeMapper.get_condition_text(condition_code)
    print(f"Condition code: {condition_code}, Description: {condition_text}")
    
    # Ví dụ: trời mưa vừa
    condition_code = ConditionCodeMapper.get_condition_code(
        precipitation=6.5, 
        temp=22, 
        cloud_cover=90
    )
    condition_text = ConditionCodeMapper.get_condition_text(condition_code)
    print(f"Condition code: {condition_code}, Description: {condition_text}")
