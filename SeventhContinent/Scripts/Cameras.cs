using UnityEngine;
using System.Collections;

public class Cameras : MonoBehaviour {

    private Vector3 startingPos;
    private Quaternion startingRot;
    public float transitionDuration = 2.5f;
    public Transform target;

    private Transform mainCam;
    private bool zoomedOut;
    private bool transitionInProcess;

    // Use this for initialization
    void Start() {
        // for isometric view/scenic view transition
        zoomedOut = false;
        mainCam = Camera.main.transform;
        startingPos = mainCam.position;
        startingRot = mainCam.rotation;
        transitionInProcess = false;
    }

    // Update is called once per frame
    void Update() {
        // for isometric view/scenic view transition
        if (Input.GetKeyUp("space") && !transitionInProcess) {
            StartCoroutine(Transition());
        }

        // camera movable only in isometric view
        if (!zoomedOut) {
            // for camera movement
            float mousePosX = Input.mousePosition.x;
            float mousePosY = Input.mousePosition.y;
            int scrollDistance = 5;
            float scrollSpeed = 70;
            if (mousePosX < scrollDistance) {
                transform.Translate(Vector3.right * -scrollSpeed * Time.deltaTime);
            }

            if (mousePosX >= Screen.width - scrollDistance) {
                transform.Translate(Vector3.right * scrollSpeed * Time.deltaTime);
            }

            if (mousePosY < scrollDistance) {
                transform.Translate(transform.forward * -scrollSpeed * Time.deltaTime);
            }

            if (mousePosY >= Screen.height - scrollDistance) {
                transform.Translate(transform.forward * scrollSpeed * Time.deltaTime);
            }

            //Scrolling Zoom  (only works when there is no building's that has hasPlaced status false (IOW building is being placed)
            // forward
            if (Input.GetAxis("Mouse ScrollWheel") < -0 && mainCam.GetComponent<BuildingPlacement>().hasPlaced) {
                if (Camera.main.orthographicSize < 30) {
                    Camera.main.orthographicSize += 5f;
                }
            }
            // back
            if (Input.GetAxis("Mouse ScrollWheel") > -0 && mainCam.GetComponent<BuildingPlacement>().hasPlaced) {
                if (Camera.main.orthographicSize > 10) {
                    Camera.main.orthographicSize -= 5f;
                }
            }

        }
    }


    IEnumerator Transition() {
        transitionInProcess = true;
        float t = 0.0f;
        //Quaternion targetRotation;
        //Vector3 dir;
        if (!zoomedOut) {
            startingPos = mainCam.position;
            startingRot = mainCam.rotation;
            Camera.main.orthographic = false;
            // these two lines does not work as intended
            // dir = -(target.position - startingPos);
            // targetRotation = Quaternion.LookRotation(dir);
            while (t < 1.0f) {
                t += Time.deltaTime * (Time.timeScale / transitionDuration);
                mainCam.position = Vector3.Lerp(startingPos, target.position, t);
                mainCam.rotation = Quaternion.Lerp(mainCam.rotation, target.rotation, Time.deltaTime);
                yield return 0;
            }
            Debug.Log("transition complete!");


            // go back to previous saved mainCam position (isometric view).
        } else {
            /*
            while (t < 1.0f) {
                t += Time.deltaTime * (Time.timeScale / transitionDuration);
                mainCam.position = Vector3.Lerp(target.position, startingPos, t);
                mainCam.rotation = Quaternion.Lerp(mainCam.rotation, startingRot, Time.deltaTime);
                yield return 0;
            }
            */
            mainCam.position = startingPos;
            mainCam.rotation = startingRot;
            Camera.main.orthographic = true;
        }
        zoomedOut = !zoomedOut;
        transitionInProcess = false;
    }
}



/*
 *      public class RTSCamera : MonoBehaviour
     {
     
     // Use this for initialization
     void Start () {
     
     }
     
     // Update is called once per frame
     void Update () {
     
     
         float mousePosX = Input.mousePosition.x; 
         float mousePosY = Input.mousePosition.y; 
         int scrollDistance = 5; 
         float scrollSpeed = 2 * Camera.main.orthographicSize + 2;
         Vector3 aPosition = new Vector3(0, 0, 0);
         float ScrollAmount = scrollSpeed *Time.deltaTime;
         const float orthographicSizeMin = 15f;
         const float orthographicSizeMax = 256f;
             
     
             //mouse left
         if ((mousePosX < scrollDistance) && (transform.position.x > -240))
              { 
               transform.Translate (-ScrollAmount,0,0, Space.World); 
              } 
             //mouse right
         if ((mousePosX >= Screen.width - scrollDistance) && (transform.position.x < 240))
              { 
                transform.Translate (ScrollAmount,0,0, Space.World);  
              }
             //mouse down
         if ((mousePosY < scrollDistance) && (transform.position.z > -240))
              { 
               transform.Translate (0,0,-ScrollAmount, Space.World); 
              } 
             //mouse up
         if ((mousePosY >= Screen.height - scrollDistance) && (transform.position.z < 240))
              { 
               transform.Translate (0,0,ScrollAmount, Space.World); ; 
              }
             //Keyboard controls 
         if ((Input.GetKey(KeyCode.UpArrow)) && (transform.position.z < 240))
              { 
               transform.Translate (0,0,ScrollAmount, Space.World); ; 
              } 
         if ((Input.GetKey(KeyCode.DownArrow)) && (transform.position.z > -240))
              { 
               transform.Translate (0,0,-ScrollAmount, Space.World);
              }
         if ((Input.GetKey(KeyCode.LeftArrow)) && (transform.position.x > -240))
              { 
               transform.Translate (-ScrollAmount,0,0, Space.World);  
              } 
         if ((Input.GetKey(KeyCode.RightArrow)) && (transform.position.x < 240))
              { 
               transform.Translate (ScrollAmount,0,0, Space.World); 
              }
             //Scrolling Zoom
           if (Input.GetAxis("Mouse ScrollWheel") < -0) // forward
                 {
            Camera.main.orthographicSize *= 1.1f;
                 }
         if (Input.GetAxis("Mouse ScrollWheel") > -0) // back
             {
            Camera.main.orthographicSize *= 0.9f;
                 }
     
         Camera.main.orthographicSize = Mathf.Clamp(Camera.main.orthographicSize, orthographicSizeMin, orthographicSizeMax );
     }
     }

    */