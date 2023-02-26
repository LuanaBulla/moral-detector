from moral_detector import MoralDetector

def main():
  detector = MoralDetector()
  txt = input(str)
  return detector.execution(txt)
  
if __name__ == "__main__":
    print(main())

