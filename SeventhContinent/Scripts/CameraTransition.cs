using UnityEngine;
using System.Collections;

public class CameraTransition : MonoBehaviour {
    /*
    Transform posCam1;
    Transform posCam2;
    Transform positionCurrent;
    GameObject transitionCamera;
    float speedAdjust;
    */

    private Vector3 startingPos;
    private Quaternion startingRot;
    public float transitionDuration = 2.5f;
    public Transform target;

    private Transform mainCam;
    private bool zoomedOut;

    public RectTransform panel;
    private Transform canvas;
    private int canvasUIsCount;

    // Use this for initialization
    void Start () {
        zoomedOut = false;
        mainCam = Camera.main.transform;
        startingPos = mainCam.position;
        startingRot = mainCam.rotation;

        //canvas = GameObject.Find("Canvas").transform;
        //canvasUIsCount = canvas.childCount;
    }
	
	// Update is called once per frame
	void Update () {
        if (Input.GetKeyUp("space")) {
            Transition();
        }
    }

    /*
    void ReOrient() {
        positionCurrent.transform.position = Vector3.Lerp(posCam1.transform.position, posCam2.transform.position, Time.deltaTime * speedAdjust);
        transitionCamera.transform.position = positionCurrent.transform.position;
    }
    */

    IEnumerator Transition() {
        //panelsOff();
        float t = 0.0f;
        Quaternion targetRotation;
        Vector3 dir;
        if (!zoomedOut) {
            startingPos = mainCam.position;
            startingRot = mainCam.rotation;
            dir = target.position - startingPos;
            targetRotation = Quaternion.LookRotation(dir);
            while (t < 1.0f) {
                t += Time.deltaTime * (Time.timeScale / transitionDuration);
                mainCam.position = Vector3.Lerp(startingPos, target.position, t);
                mainCam.rotation = Quaternion.Lerp(mainCam.rotation, targetRotation, Time.deltaTime);
                yield return 0;
            }
            // activate this panel
            //panel.gameObject.SetActive(true);

        // go back to previous saved mainCam position (isometric view).
        } else {
            while (t < 1.0f) {
                t += Time.deltaTime * (Time.timeScale / transitionDuration);
                mainCam.position = Vector3.Lerp(target.position, startingPos, t);
                mainCam.rotation = Quaternion.Lerp(mainCam.rotation, startingRot, Time.deltaTime);
                yield return 0;
            }
            // activate main panel
            //canvas.GetChild(1).gameObject.SetActive(true);
        }
        zoomedOut = !zoomedOut;
    }

    /*
    void OnMouseUp() {
        StartCoroutine(Transition());
    }
    */

    /*
    void panelsOff() {
        for (int i = 1; i < canvasUIsCount; i++) {
            canvas.GetChild(i).gameObject.SetActive(false);
        }
    }
    */
}
