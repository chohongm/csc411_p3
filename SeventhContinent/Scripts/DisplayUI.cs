using UnityEngine;
using UnityEngine.UI;
using System.Collections;

public class DisplayUI : MonoBehaviour {

    //public string myString;
    public Text myText;
    //private Text myText;
    public float fadeTime;
    public bool displayInfo;
    //public Camera cameraToLookAt;

    // Use this for initialization
    void Start() {
        //myText = textGO.GetComponentInChildren<Text>();
        myText.color = Color.clear;
        //Screen.showCursor = false;
        //Screen.lockCursor = true;
        /*
        if (cameraToLookAt != null) {
            Vector3 v = cameraToLookAt.transform.position - myText.transform.position;
            v.x = v.z = 0.0f;
            myText.transform.LookAt(cameraToLookAt.transform.position - v);
            myText.transform.Rotate(0, 180, 0);
            //myText.transform.rotation = Quaternion.LookRotation(Camera.main.transform.forward, Camera.main.transform.up);
        }
        */
    }

    // Update is called once per frame
    void Update() {
        FadeText();
        /*if (Input.GetKeyDown (KeyCode.Escape)) 
         
                {
                        Screen.lockCursor = false;
                         
                }
                */
    }

    public void FadeIn() {
        displayInfo = true;

    }

    public void FadeOut() {
        displayInfo = false;
    }

    void FadeText() {
        if (displayInfo) {
            //myText.text = myString;
            myText.color = Color.Lerp(myText.color, Color.black, fadeTime * Time.deltaTime);
        } else {
            myText.color = Color.Lerp(myText.color, Color.clear, fadeTime * Time.deltaTime);
        }
    }
}