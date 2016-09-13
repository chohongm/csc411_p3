using UnityEngine;
using System.Collections;

public class Cart : MonoBehaviour {

    public GameObject pathGO;

    Transform targetPathNode;
    int pathNodeIndex = 0;
    int pathSize;
    float speed = 5f;

	// Use this for initialization
	void Start () {
        //pathGO = GameObject.Find("Cart Path");
        pathSize = pathGO.transform.childCount;
    }
	
    void GetNextPathNode() {
        targetPathNode = pathGO.transform.GetChild(pathNodeIndex);
        pathNodeIndex++;
    }
	// Update is called once per frame
	void Update () {
        if (pathGO != null) {
            if (targetPathNode == null) {
                if (pathNodeIndex < pathSize) {
                    GetNextPathNode();
                } else {
                    ReachedGoal();
                }
            }
            Vector3 dir = targetPathNode.position - this.transform.localPosition;
            float distThisFrame = speed * Time.deltaTime;
            if (dir.magnitude <= distThisFrame) {
                // we reached the node
                targetPathNode = null;
            } else {
                // Move towards node
                transform.Translate(dir.normalized * distThisFrame, Space.World);
                Quaternion targetRotation = Quaternion.LookRotation(dir);
                this.transform.rotation = Quaternion.Lerp(this.transform.rotation, targetRotation, Time.deltaTime * 5);
            }
        }
	}

    void ReachedGoal() {
        Destroy(gameObject);
    }
}
