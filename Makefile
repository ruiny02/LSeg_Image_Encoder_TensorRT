# Makefile (프로젝트 루트에 위치)

.PHONY: all bench feature clean

# 기본 타겟: 벤치 + 피처 익스트랙터 빌드
all: bench feature

# 1) C++ 벤치마크 빌드
bench:
	@echo "🔨 Building Inference_Time_Tester..."
	@rm -rf CPP_Project/Inference_Time_Tester/build
	@mkdir -p CPP_Project/Inference_Time_Tester/build
	@cd CPP_Project/Inference_Time_Tester/build && cmake .. && make -j$$(nproc)

# 2) C++ Feature Extractor 빌드
feature:
	@echo "🔨 Building Feature_Extractor..."
	@rm -rf CPP_Project/Feature_Extractor/build
	@mkdir -p CPP_Project/Feature_Extractor/build
	@cd CPP_Project/Feature_Extractor/build && cmake .. && make -j$$(nproc)

# 3) 클린: 모든 build 폴더 삭제
clean:
	@echo "🧹 Cleaning all build directories..."
	@rm -rf CPP_Project/Inference_Time_Tester/build
	@rm -rf CPP_Project/Feature_Extractor/build
