from moral_detector import MoralDetector

def main():
  detector = MoralDetector()
  txt = input(str)
  print(detector.execution(txt))
  
if __name__ == "__main__":
    main()

